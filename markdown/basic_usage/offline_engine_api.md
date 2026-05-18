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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.54it/s]


    2026-05-18 16:57:02,900 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 16:57:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:42,  3.90s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.66it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.93it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.93it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.96it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.48it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.11it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.47it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.41it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.82it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.09it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 37.68it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 37.68it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.10it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.10it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.10it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.10it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.10it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.16it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.03it/s]

    Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.34it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.87it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 43.55it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 37.69it/s]


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
    Generated text:  Sam and I am 23 years old. I am American and I live in Atlanta, Georgia. I am a vegetarian and I do not eat meat, dairy, or eggs. I love to spend my time reading and traveling. I am currently living with my partner, Nick, who is also an American and we live in Atlanta. We have been married for 7 years and have had our first child. I had a heart attack 3 years ago and have a BMI of 25. 
    
    What can I do to stay healthy? What should I eat? What should I not eat? How should I travel? How should
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to get the united states to join a new country. If the president were to meet with a new country, what kind of country might the president meet with? Choose the most suitable option to answer the above question. Options: - mandarin ducks - peacock - hen house - cave - country
    The answer to this problem is:
    country.
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Moscow D. Paris D. Paris
    
    The capital of France is Paris. It is known as the "City of Love" due to its romantic architecture and beautiful parks. Paris is a popular tourist destination known for its famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral. It is also a cultural and intellectual center in Europe, known for its art, music, and fashion. As a result, Paris is often considered the "City of Love" and is a favorite among both romantic and artistic individuals.
    
    Other options such as London or Moscow are not capital
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be marked by the integration of artificial intelligence in all aspects of society. As the technology evolves, it is expected to lead to a more sustainable and efficient future. However, there are also challenges to consider, including the need for ethical oversight and the potential for AI to be used for nefarious purposes. It is important to carefully consider the potential risks and benefits of AI in order to ensure that it is developed and used in a responsible and ethical manner.
    To that end, here are some key points to consider when evaluating the potential risks and benefits of AI:
    
    Potential risks of AI include:
    
    1. Bias and discrimination: AI algorithms can


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your profession or skills]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. I like to read, watch movies, and listen to music. What's your favorite hobby? I like to [insert a short
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city with a rich history and diverse culture. It is located in the south of France and is the largest city in the country, with a population of over 1.3 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its vibrant nightlife and food scene. The city is also home to many world-renowned museums, including the Louvre and the Musée d'Orsay. Paris is a cultural and economic hub of France and a major tourist destination. It is also known
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more widespread use of AI in healthcare,
    


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
    Generated text:  [First Name] and I'm a [Last Name] from [Your city or state]. I'm a [job title] with [number of years in the industry]. I've been with my company for [number of years] and I love [something about your job or industry].
    My goal is to be a valuable member of your team, and I'm excited to bring my energy, passion, and knowledge to every project. I'm looking forward to the opportunity to share my skills and experience with you, and I'm confident that I can make a positive impact on your team. What's your name, and what's your role
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the most populous city in the European Union.
    
    -1933-2022
    
    Translate the French word "Capitale" to English.
    
    -1933-2022
    
    Translate the following sentence into English: "I need a new car for my work". Translate the following sentence into English: "I need a new car for my work".
    
    Translate the following sentence into English: "I need to work at home from now on". Translate the following sentence into English: "I need to work at home from now on".
    
    Translate the following sentence
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a combination of current trends and new innovations. Here are some potential areas of focus:
    
    1. Natural Language Processing: With the rise of big data and AI, the ability to process and understand natural language has become more advanced. This will lead to more sophisticated language-based AI systems, which will be able to understand and respond to human language in new ways.
    
    2. Image and Speech Recognition: While image and speech recognition technologies have been around for a while, their accuracy and precision are improving rapidly. Future developments could bring even more advanced recognition capabilities.
    
    3. Autonomous Vehicles: Self-driving cars are likely to become more common


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

    ].

     I

    'm

     a

     [

    Age

    ]

     year

     old

     AI

     assistant

    ,

     programming

     to

     learn

     and

     improve

     my

     ability

     to

     provide

     helpful

     responses

    .

     I

     am

     here

     to

     help

     you

     with

     any

     questions

     you

     might

     have

     about

     the

     world

     and

     technology

    ,

     as

     well

     as

     to

     assist

     in

     any

     tasks

     you

     might

     need

     assistance

     with

    .

     Please

     feel

     free

     to

     ask

     me

     anything

     and

     I

    'll

     do

     my

     best

     to

     provide

     you

     with

     accurate

     and

     helpful

     responses

    .

     Thank

     you

     for

     having

     me

    .

     [

    Name

    ]

     How

     can

     I

     help

     you

     today

    ?

     [

    Name

    ]

     Thanks

     for

     having

     me

    .

     How

     can

     I

     help

     you

     today

    ?

     [

    Name

    ]

     Hi

     there

    !

     How

     are

     you

    ?

     I

    'm

     an

     AI

     assistant

    ,

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     has

     been

     a

     major

     European

     city

     since

     the

     

    1

    2

    th

     century

    .

     
    


    **

    Explanation

     of

     the

     statement

    :**

     The

     statement

     is

     a

     factual

     representation

     of

     the

     capital

     city

     of

     France

    .

     It

     con

    veys

     that

     the

     capital

     is

     Paris

    ,

     a

     city

     known

     for

     being

     a

     significant

     European

     location

     since

     the

     

    1

    2

    th

     century

    .

     The

     mention

     of

     the

     "

    City

     of

     Light

    "

     indicates

     that

     it

     is

     renowned

     for

     its

     cultural

     and

     artistic

     heritage

    ,

     which

     has

     been

     a

     driving

     force

     behind

     its

     global

     prominence

     and

     influence

    .

     Additionally

    ,

     the

     term

     "

    City

     of

     Light

    "

     refers

     to

     its

     ability

     to

     attract

     visitors

     from

     across

     the

     globe

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     going

     to

     be

     impacted

     by

     several

     trends

     and

     technologies

     that

     are

     set

     to

     continue

     shaping

     its

     development

    .

     Here

     are

     some

     of

     the

     most

     promising

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     will

     become

     more

     pervasive

    :

     As

     AI

     systems

     become

     more

     widely

     used

    ,

     we

     will

     see

     a

     greater

     penetration

     of

     AI

     in

     various

     sectors

    ,

     including

     healthcare

    ,

     finance

    ,

     transportation

    ,

     and

     entertainment

    .

     This

     will

     lead

     to

     increased

     automation

     and

     optimization

     in

     these

     industries

    .
    


    2

    .

     Machine

     learning

     will

     become

     more

     sophisticated

    :

     Machine

     learning

     is

     the

     key

     to

     AI

    's

     future

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     will

     see

     more

     advanced

     models

     that

     can

     learn

     from

     data

     and

     improve

     their

     performance

     over

     time

    .

     This

     will

     lead

     to

    



```python
llm.shutdown()
```
