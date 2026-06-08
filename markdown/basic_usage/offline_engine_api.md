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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.86it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.95it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.95it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 31.89it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.37it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 18.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.61it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.20it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  50%|█████     | 29/58 [00:00<00:00, 42.91it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.51it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=112 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 39.17it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 40.75it/s]Capturing num tokens (num_tokens=20 avail_mem=74.55 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=16 avail_mem=74.55 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.73it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  93%|█████████▎| 54/58 [00:01<00:00, 41.73it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 39.20it/s]


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
    Generated text:  Alessandro. I'm a computer science graduate student at the University of Arizona, focusing on AI for education.
    
    I'm interested in a variety of topics, but in particular I'm fascinated by the intersection of machine learning and the use of technology in educational settings. I'm eager to learn more about the latest developments in these fields and how they're shaping the future of education. What are some of the most exciting developments you're particularly interested in learning about? Let me know if you'd like to know more! 
    
    [Markdown]
    ```python
    # Define the variables for the question
    interest1 = "the intersection of machine learning and the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president is like the boss of the whole country. There are 13 members in a country called the Senate, so there are 13 members of the Senate. Each member of the Senate has a job. The members of the Senate have different jobs, but the president has a job too. There are 13 jobs in the Senate, and the president has one job. The president has a job like being the boss of the Senate. The president has a job where he can talk to the country. He can talk to the people of the country. The president has a job that he can use
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ).
    A. Paris
    B. Strasbourg
    C. Brussels
    D. Nice
    Answer:
    
    A
    
    When the interest rate is known, the relationship between the effective annual rate, the nominal annual rate, and the time is ( ).
    A. The effective annual rate equals the nominal annual rate multiplied by the number of compounding periods per year
    B. The effective annual rate is the difference between the nominal annual rate and the number of compounding periods per year
    C. The effective annual rate is the difference between the nominal annual rate and the number of compounding periods per year
    D. The effective annual rate is equal to
    ===============================
    Prompt: The future of AI is
    Generated text:  moving towards a real-time learning framework, where a learner uses the machine learning model to learn from the data it receives. However, to ensure that the real-time learning framework works effectively, a learner needs to be able to understand and interpret the output of the model. This requires a deep understanding of the model's architecture, data, and the inputs it receives. In addition, it requires the ability to analyze the model's performance and make necessary adjustments to improve its accuracy.
    To address this challenge, we can use a combination of methods such as data exploration, feature engineering, model evaluation, and hyperparameter tuning. Data exploration involves understanding the data


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is known for its cuisine, including its famous French fries and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that has played a significant role in French history and continues to be a major cultural and economic center in the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, privacy, and transparency. AI developers will need to be more mindful of the potential impact of their work on society and will need to take steps to ensure that their algorithms are fair and unbiased.
    
    2. Greater integration with other technologies: AI is already being integrated
    


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
    Generated text:  [Name], and I'm a [Type] [Occupation]. [Name] is a unique individual who brings a fresh perspective and creative flair to any project I work on. With a natural talent for problem-solving and a strong sense of humor, [Name] excels in creating innovative ideas and designing functional solutions. Whether it's a new product, a new marketing campaign, or a novel creative concept, [Name] is a must-have in any project or endeavor. [Name] is a passionate and kind-hearted individual who thrives in the creative environment and is always looking for new challenges to take me on. [Name] is
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the second most populous and most important city in France, with a population of over 10 million people. It is the largest city in the European Union and the largest city outside of Europe. The city is home to many famous landmarks and attractions, including the Eiffel Tower, the Louvre Museum, the Notre-Dame Cathedral, and the Champs-Élysées. Paris is known for its rich cultural heritage, culinary traditions, and music scene, and is a major center for international business, fashion, and culture. The city has a diverse and multicultural population, with over 200 languages spoken
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and filled with possibilities. Here are some potential future trends that will shape the industry:
    
    1. Increased integration with other technologies: AI will continue to blend seamlessly with other emerging technologies such as blockchain, IoT, and quantum computing. This integration will bring new efficiencies and applications for AI, as these technologies will become more prevalent and integrate with AI to create even more advanced solutions.
    
    2. AI will become more sophisticated: Over the next decade, we can expect AI to become even more sophisticated, with the ability to learn from experience, adapt to new situations, and understand more complex problems. This will result in more accurate predictions, better decision-making


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

     [

    Type

     of

     Person

    ]

     by

     profession

    .

     I

    'm

     a

     [

    Occup

    ation

    ]

     with

     a

     [

    Skill

    /

    Tool

    ]

     I

    'm

     very

     [

    Cur

    iosity

    /

    Ad

    m

    iration

    /

    En

    thus

    iasm

    ]

     about

    .

     I

    'm

     [

    Your

     Unique

     Selling

     Point

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     [

    Skill

    /

    Tool

    ].

     What

     kind

     of

     experiences

     do

     you

     like

     to

     have

    ?

     I

     enjoy

     [

    Favorite

     Activity

    /

    Event

    /

    Experience

    ],

     and

     I

     love

     [

    Favorite

     Food

    /

    Drink

    ].

     What

     are

     your

     hobbies

    ?

     I

     like

     [

    H

    obby

     

    1

    ],

     [

    H

    obby

     

    2

    ],

     and

     [

    H

    obby

     

    3

    ].

     I

    'm

     also

     an

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     officially

     the

     city

     of

     Paris

    ,

     is

     the

     capital

     and

     most

     populous

     city

     of

     France

    .

     Located

     in

     the

     north

     central

     part

     of

     the

     country

     and

     bounded

     by

     the

     Se

    ine

     river

     to

     the

     west

     and

     south

    ,

     the

     city

     of

     Paris

     has

     a

     population

     of

     around

     

    2

    .

     

    1

     million

     and

     is

     the

     most

     cosm

    opolitan

     of

     France

    's

     three

     major

     cities

    .

     The

     city

     was

     founded

     on

     September

     

    1

    ,

     

    7

    8

    9

     by

     French

     colon

    ists

    ,

     and

     has

     become

     a

     UNESCO

     world

     heritage

     site

    .

     The

     city

    's

     architecture

     is

     characterized

     by

     its

     orn

    ate

     architecture

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     and

     its

     influence

     on

     French

     design

    .

     The

     city

     is

     a

     center

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     exciting

     and

     unpredictable

    ,

     with

     a

     range

     of

     potential

     trends

     that

     are

     shaping

     the

     landscape

     of

     the

     technology

     industry

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     use

     of

     AI

     for

     natural

     language

     processing

    :

     As

     more

     people

     use

     voice

     assistants

     like

     Siri

    ,

     Alexa

    ,

     and

     Google

     Assistant

    ,

     there

     will

     be

     an

     increasing

     demand

     for

     AI

    -powered

     translation

    ,

     speech

     recognition

    ,

     and

     other

     natural

     language

     processing

     tools

    .

     This

     will

     drive

     further

     investment

     in

     AI

     research

     and

     development

    .
    


    2

    .

     AI

     in

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

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     As

     the

     technology

     continues

     to

     advance

    ,

     we

     can

    



```python
llm.shutdown()
```
