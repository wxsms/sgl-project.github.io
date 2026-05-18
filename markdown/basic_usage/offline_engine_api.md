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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.05it/s]


    2026-05-18 02:42:02,450 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 02:42:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:19,  4.55s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.43it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.27it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.27it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.42it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.42it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.42it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.42it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.42it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.32it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 17.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.58it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.58it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.72it/s] Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]

    Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.00it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.45it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 42.16it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.50it/s] Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.67it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.80it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 38.57it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 36.96it/s]


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
    Generated text:  Tobias, I am 19 years old. I am a software developer, always learning and always asking questions. I am looking for a job. What are my strengths, weaknesses and what kind of companies do you like to work for?
    Sure, I'd be happy to help! Let's start by discussing your strengths and weaknesses, and then I'll try to figure out what kind of companies you might like to work for. Can you tell me a little bit about yourself? What programming languages do you prefer? What kind of software do you write? Do you have experience with certain types of software, or do you prefer to work with
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a third term in office. He is running against a incumbent president who was previously the president for two terms and is not running for a third term. The two candidates will speak to the public for the last time before the last day of voting. The first candidate has to speak for 30 minutes, the second candidate has to speak for 40 minutes, and the third candidate has to speak for 50 minutes. The second candidate is planning to take a 5-minute break after the first 30 minutes, and the third candidate is planning to take a 7-minute break after the first 40 minutes
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It has a population of approximately 2, 100, 000. The name Paris is the French version of the English "Parker" meaning "wood."
    
    Since the French Revolution of 1789, Paris has been the political, financial, and cultural centre of France. It has been the seat of government, administration, commerce, and industry since the French Revolution. It is also the largest city in France, with a city centre of 300, 000 square metres (300 acres).
    
    It has been home to both French and international artists. The city hosts
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting one, but it also brings up some important ethical questions. What are the most pressing issues related to AI, and how do they impact society as a whole?
    AI has the potential to revolutionize many aspects of life, from healthcare and education to transportation and security. However, the ethical implications of AI are still emerging and are critical to consider.
    One of the most pressing issues related to AI is the issue of bias. AI systems can be trained on large datasets that may contain biases, which can lead to discriminatory outcomes. For example, a facial recognition system trained on biased images may identify individuals of a particular race or ethnicity as suspects


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill] who has been [Number of Years] years in the industry. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [Favorite Hobby] and I enjoy [What I Do for Fun]. I'm always looking for ways to improve myself and make the world a better place. I'm excited to meet you and learn more about you. [Name] [Age] [Occupation] [Skill] [Favorite Hobby] [What I Do for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the arts and culture of the world. It is also a popular tourist destination, with millions of visitors annually. The city is home to many famous landmarks and attractions, including the Palace of Versailles, the Arc de Triomphe, and the Champs-Élys
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: As AI continues to advance, we can expect to see more and more automation in various industries, including manufacturing, transportation, and healthcare. This will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. This will likely lead to more stringent regulations and standards for
    


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
    Generated text:  [Name] and I'm a [occupation/curiosity] who has always been fascinated by [what you might know about your interests, hobbies, etc.]. I'm now a [position] with [company name] and I enjoy [mention a specific activity or hobby]. I’m [age range] years old and [job title or position]. I'm a [general personality trait or characteristic] person who is [general personality trait or characteristic]. I'm always [general personality trait or characteristic], which makes me a good [job type]. I'm always [general personality trait or characteristic], which makes me a good [job type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and is home to the Eiffel Tower, the Louvre Museum, and many other attractions. The city is known for its historical architecture and vibrant cultural scene. Paris is also famous for its artistic and literary traditions, including the annual Les Misérables Festival. The city has a rich history dating back to the Roman Empire, and is home to the iconic Eiffel Tower that stands tall over the city. Paris is a hub for the arts, and many famous artists, writers, and musicians have originated from or settled in the city. Overall, Paris is a cultural and urban
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking very promising, with many possibilities and exciting developments ahead. Here are some potential future trends in AI:
    
    1. Personalized AI: AI will become more personalized and more effective in delivering customized solutions to individual users. This could involve AI algorithms that can learn from data about an individual's behavior and preferences to provide personalized recommendations and recommendations.
    
    2. Autonomous vehicles: Autonomous vehicles (AVs) are expected to become a major part of everyday life. AI technology will be used to develop intelligent and self-driving cars that can navigate roads safely and make decisions based on real-time data.
    
    3. Healthcare: AI will play a critical role in the


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

     a

     seasoned

     [

    field

     or

     profession

    ]

     with

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    job

     title

    ].

     I

     am

     [

    positive

     and

     enthusiastic

     about

     my

     work

     style

    ]

     and

     I

     am

     always

     looking

     to

     learn

     and

     improve

    .

     I

     am

     confident

     in

     my

     ability

     to

     [

    describe

     a

     specific

     skill

     or

     accomplishment

     that

     demonstrates

     my

     ability

    ]

     and

     I

     am

     eager

     to

     [

    describe

     how

     I

     plan

     to

     continue

     my

     growth

     and

     development

    ].

     I

     am

     always

     ready

     to

     listen

     and

     collaborate

     with

     others

    ,

     and

     I

     am

     a

     great

     communicator

    .

     I

     am

     a

     [

    person

    ality

     type

    ]

     and

     I

     am

     always

     seeking

     to

     make

     a

     positive

     impact

     and

     help

     others

    .

     [

    Person

    ality

     trait

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     an

     iconic

     city

     known

     for

     its

     historic

     landmarks

    ,

     lively

     nightlife

    ,

     and

     rich

     cultural

     scene

    .

     It

     was

     founded

     in

     the

     

    1

    2

    th

     century

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     cafes

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     Paris

     is

     also

     known

     for

     its

     art

    ,

     music

    ,

     and

     cuisine

    ,

     with

     numerous

     festivals

     and

     events

     throughout

     the

     year

    .

     In

     addition

     to

     being

     a

     major

     city

    ,

     Paris

     is

     also

     a

     significant

     economic

     and

     political

     center

    ,

     hosting

     major

     international

     organizations

     and

     hosting

     many

     famous

     works

     of

     art

     and

     architecture

    .

     The

     city

     is

     known

     for

     its

     romantic

     architecture

    ,

     picturesque

     countryside

    ,

     and

     diverse

     cultural

     influences

    .

     Paris

     has

     a

     strong

     sense

     of

     history

     and

     identity

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     advancements

     in

     several

     key

     areas

    :
    


    1

    .

     Increased

     Personal

    ization

    :

     As

     AI

     continues

     to

     improve

    ,

     it

     is

     expected

     to

     become

     more

     personalized

     and

     adaptable

    .

     This

     will

     allow

     for

     more

     precise

     and

     efficient

     solutions

     to

     various

     problems

    ,

     as

     well

     as

     a

     wider

     range

     of

     applications

     that

     are

     tailored

     to

     specific

     individuals

     and

     contexts

    .
    


    2

    .

     Autonomous

     machines

    :

     Autonomous

     machines

     will

     become

     more

     common

     as

     AI

     technology

     advances

    ,

     allowing

     them

     to

     make

     decisions

     without

     human

     intervention

    .

     This

     will

     enable

     more

     efficient

     and

     precise

     solutions

     to

     problems

    ,

     as

     well

     as

     the

     ability

     to

     interact

     with

     the

     outside

     world

    .
    


    3

    .

     Machine

     learning

    :

     AI

     will

     continue

     to

     improve

     its

     ability

     to

     learn

     from

     data

    



```python
llm.shutdown()
```
