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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:16,  4.51s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.31it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.31it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.65it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.53it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.53it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.53it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.53it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.53it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.48it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.22it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.12it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.58it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.75it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.04it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:00<00:00, 43.32it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.32it/s]

    Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s]Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.66it/s] Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]

    Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  81%|████████  | 47/58 [00:01<00:00, 45.25it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 45.65it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.13it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 46.13it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.32it/s]


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
    Generated text:  John. I'm a middle school student. I like drawing. I like to go to the countryside. I love singing and dancing. I often go to the movies. I like my parents, but I don't like my brother. I like Chinese, but my brother likes English. I like to read. I like to learn something new. I like to have fun. I like to go to the park with my friends. I like to eat good food. I like to have a good sleep. What's the best title for this passage? A) My Family B) My School C) My Life D) My Tutors D
    ===============================
    Prompt: The president of the United States is
    Generated text:  retiring after a 25-year career in government. He is considering giving the office to his son-in-law, who is also a presidential candidate, or selling the office to the highest bidder. He has received a donation of $50 million from a wealthy individual, which he plans to invest in the stock market. The stock market is expected to grow at a rate of 10% per year. If he decides to sell the office immediately, he will receive $100 million in cash. However, if he keeps the office, he will receive a bonus of $50 million in cash and a three-year bonus of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A.正确
    B.错误
    答案:
    
    A
    
    The airport is far from the city.
    A.正确
    B.错误
    答案:
    
    A
    
    The capital of France is Paris. The airport is far from the city.
    A.正确
    B.错误
    答案:
    
    A
    
    I don't think he will help me.
    A.正确
    B.错误
    答案:
    
    B
    
    You didn't say anything.
    A.正确
    B.错误
    答案:
    
    B
    
    We should not eat too much.
    A.正确
    B.错误
    答案:
    
    B
    
    We should not smoke.
    A.正确
    
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain, as concerns about its ethical implications remain unaddressed. Here’s how AI technology is influencing the field.
    
    The future of AI is uncertain, as concerns about its ethical implications remain unaddressed. Here’s how AI technology is influencing the field.
    
    ### AI and the Future of Work
    
    Artificial intelligence (AI) has already transformed the way businesses function and has the potential to revolutionize how businesses operate. AI has the potential to automate many of the mundane tasks that humans perform, such as scheduling, data entry, and simple administrative work, which has reduced the need for human intervention.
    
    Moreover, AI has the potential to be used


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name]. I've been driving for [Number of Years] years and have [Number of Miles] miles on my [Vehicle Name]. I'm a [Favorite Hobby] [Hobby Name] and I enjoy [Favorite Activity] [Activity Name]. I'm a [Favorite Person] [Person Name] and I have [Number of Friends] friends. I'm a [Favorite Book] [Book Title] and I love [Favorite Food] [Food Name]. I'm a [Favorite Movie] [Movie
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic and culturally rich city located in the south of the country. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. The city is also famous for its fashion industry, cuisine, and music scene. Paris is a vibrant and diverse city with a rich history and a strong sense of identity. It is a popular tourist destination and a major economic and cultural center in France. The city is home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation and efficiency: AI is expected to continue to automate a wide range of tasks, from manufacturing and transportation to customer service and healthcare. This will lead to increased efficiency and productivity, as machines can perform tasks that were previously done by humans.
    
    2. Enhanced human-machine collaboration: AI will continue to improve its ability to understand and interact with humans, leading to more seamless and efficient collaboration between humans and machines. This will require greater understanding of human emotions, motivations, and behaviors, as well as the ability to adapt to changing circumstances.
    
    3. AI-driven innovation: AI will continue to
    


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
    Generated text:  [Name] and I'm an AI language model. I'm programmed to assist users in generating natural language responses and answering their questions. With my advanced algorithms and machine learning techniques, I'm capable of generating responses that are not only accurate but also helpful, providing a friendly and friendly user experience. What's your name, and what can you do for me today? The key to becoming a successful AI language model is continuous learning and improvement. I constantly learn from new data and updates to improve my responses and accuracy. To help you understand how I work, here's an example response: "Hello! How can I assist you today?" And
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city that has been the seat of the government and the center of French culture for over 400 years. Located in the heart of the French region of the Île-de-France, the city is known for its iconic Eiffel Tower, la Gare de l’Est and the beautiful Palais des Nations. It is a bustling metropolis with a rich history and a reputation for being a beautiful, livable city. Paris is also the largest city in France by population. The city is home to over 3 million people and is the third-largest metropolitan area in the world. Its status as the capital
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid innovation and the development of new technologies that will enable AI to perform increasingly complex tasks and operate in more diverse and uncertain environments. Here are some potential future trends in AI:
    
    1. Improved cognitive functions: AI will continue to improve its ability to process and analyze large amounts of data, and to learn and adapt to new situations. This will be achieved through advances in machine learning, natural language processing, and other areas of research.
    
    2. Deep learning: Deep learning is a subset of machine learning that involves the use of neural networks with many layers. Deep learning will be especially important in applications that require high levels of


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

    ],

     and

     I

     am

     a

     computer

     scientist

     who

     specializes

     in

     AI

     and

     machine

     learning

    .

     I

     have

     a

     passion

     for

     using

     my

     skills

     to

     solve

     complex

     problems

     and

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

     enjoy

     exploring

     new

     technologies

     and

     experimenting

     with

     new

     ideas

     to

     solve

     challenging

     problems

    .

     I

     am

     always

     looking

     to

     learn

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     advancements

     in

     the

     field

    .

     Thank

     you

     for

     asking

    !

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     As

     a

     computer

     scientist

    ,

     I

     specialize

     in

     AI

     and

     machine

     learning

    .

     I

     have

     a

     passion

     for

     using

     my

     skills

     to

     solve

     complex

     problems

     and

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

     enjoy

     exploring

     new

     technologies

     and

     experimenting

     with

     new

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     and

     is

     a

     historic

     and

     culturally

     rich

     city

     with

     a

     diverse

     population

     and

     impressive

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     
    


    The

     city

     is

     located

     in

     the

     Paris

     Basin

    ,

     which

     is

     a

     region

     in

     the

     south

     of

     the

     country

    .

     It

     is

     situated

     on

     the

     Left

     Bank

     of

     the

     Se

    ine

    ,

     which

     flows

     through

     the

     city

     and

     provides

     the

     city

     with

     access

     to

     the

     river

     and

     its

     famous

     bridges

    .

     The

     city

     is

     also

     known

     for

     its

     vibrant

     culture

    ,

     music

    ,

     and

     food

    .

     
    


    Paris

     is

     a

     bustling

     met

    ropolis

     with

     an

     international

     reputation

     for

     its

     fashion

    ,

     art

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     can

     be

     influenced

     by

     a

     variety

     of

     factors

    ,

     including

     technological

     advancements

    ,

     policy

     changes

    ,

     and

     societal

     shifts

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

    :

     AI

     will

     continue

     to

     evolve

     and

     become

     more

     capable

     of

     performing

     complex

     tasks

     more

     quickly

     and

     accurately

     than

     humans

    .

     This

     could

     lead

     to

     significant

     changes

     in

     employment

     and

     industries

    ,

     as

     automation

     will

     replace

     human

     workers

     in

     certain

     roles

    .
    


    2

    .

     Improved

     privacy

     and

     security

    :

     With

     AI

     systems

     becoming

     more

     powerful

    ,

     there

     will

     be

     a

     growing

     need

     to

     protect

     against

     data

     breaches

    ,

     cyber

     attacks

    ,

     and

     other

     risks

     to

     privacy

     and

     security

    .

     AI

     will

     need

     to

     be

     designed

     with

     these

     concerns

     in

     mind

    ,

     as

     well

    



```python
llm.shutdown()
```
