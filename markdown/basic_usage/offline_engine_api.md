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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.89it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.89it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.89it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.33it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   3%|▎         | 2/58 [00:00<00:02, 19.01it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.03 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.02 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):   9%|▊         | 5/58 [00:00<00:02, 22.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.00 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.99 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.98 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.02it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.97 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.95 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=960 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s] Capturing num tokens (num_tokens=896 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.98it/s]Capturing num tokens (num_tokens=832 avail_mem=70.96 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=768 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=704 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=640 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=576 avail_mem=70.95 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.20it/s]Capturing num tokens (num_tokens=512 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=480 avail_mem=70.95 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=448 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=416 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=384 avail_mem=70.94 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]

    Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  50%|█████     | 29/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=352 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=320 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=288 avail_mem=70.93 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=256 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=240 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.89it/s]Capturing num tokens (num_tokens=224 avail_mem=70.92 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.18it/s]Capturing num tokens (num_tokens=208 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.18it/s]Capturing num tokens (num_tokens=192 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=176 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=160 avail_mem=70.91 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]

    Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.18it/s]Capturing num tokens (num_tokens=144 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=128 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=112 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=96 avail_mem=70.90 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s] Capturing num tokens (num_tokens=80 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.96it/s]Capturing num tokens (num_tokens=64 avail_mem=70.89 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=48 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=32 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=28 avail_mem=70.88 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=24 avail_mem=70.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]

    Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.83it/s]Capturing num tokens (num_tokens=20 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.11it/s]Capturing num tokens (num_tokens=16 avail_mem=70.87 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.11it/s]Capturing num tokens (num_tokens=12 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.11it/s]Capturing num tokens (num_tokens=8 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.11it/s] Capturing num tokens (num_tokens=4 avail_mem=70.86 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.11it/s]Capturing num tokens (num_tokens=4 avail_mem=70.86 GB): 100%|██████████| 58/58 [00:01<00:00, 41.85it/s]


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
    Generated text:  Denise and I am an adult education instructor and the owner of a professional business. I've been here for 10 years now and I know there are a lot of people who have been using the 101 training program and that is just the beginning of what I can teach them.
    Many people have questions about the training and I've had to help them with those. I have been helping other teachers and professors for over 20 years and I know how to help them find their way around and have been teaching a variety of skills in different areas. I have a degree in Business Administration and have been working as a teacher for
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person elected by the people to represent the people. This statement is (  )
    A: Correct
    B: Incorrect
    C: Incomplete
    D: Uncertain
    To determine whether the statement "The president of the United States is a person elected by the people to represent the people" is correct, we need to analyze the core of the statement and its logical consistency.
    
    1. **Definition of a President of the United States:**
       - A president of the United States is the head of state and head of government of the United States. They are appointed by the President of the United States.
       - The term "
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, it is the most visited city in the world, and it is full of history, culture, art, and of course, delicious cuisine.
    Paris is also the home to the most exclusive hotels in the world. The owners of these expensive properties are not afraid to be provocative, but they are also prepared to take risks in order to maintain the cost of the property.
    To raise your expectations, we will follow the leading hotels in Paris, the Lidl, and the The Luxury Hotel, and we will analyze the key factors that make each hotel stand out and why they are so valuable. The purpose of this article is
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of scientists and engineers, not the minds of politicians or regulators. That is what an extensive study of the world's leading AI researchers found.
    Researchers at Stanford University, MIT, and Google University in California are mapping out the world of artificial intelligence, and they found that if the world is to move forward, it will require a shift in the way we think about the problem.
    "Google's TALK (Technology and Big Science: The State of the Art in the Internet of Things) is a four year, \$14 million initiative focused on research that will open up new frontiers in the application of big science. The


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title] at [company name], and I'm excited to be here today. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French National Museum, and the French Quarter. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. Its history dates back to the Roman Empire and has been a major center of European culture and politics for centuries. The city is known for its fashion, art, and cuisine, and is a major hub for business and commerce in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations and tasks. This could lead to more efficient and effective use of AI in various fields, such as healthcare, transportation, and manufacturing.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability.
    


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
    Generated text:  [Name] and I am [Age]. I come from [Background], but I am the [Type] version of myself, [Name]. I am known for my [Strengths and Qualities] and I have always been [Curiosity, Passion, or Other]. I am a [Occupation], and I love [Personal Hobby or Passion]. I am [What you are most proud of]. I want to encourage [Person] to [What you want them to do]. I strive to [Why you do what you do]. I am [What you are like]. What are your hobbies? I enjoy [Your hobby], [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre Dame Cathedral, and the Arc de Triomphe.
    
    What is the name of the most famous tourist attraction in Paris? The most famous tourist attraction in Paris is the Eiffel Tower, which stands as a symbol of France and Parisian culture.
    
    Please provide the name of the French President who has lived in Paris for the longest period. The French President who has lived in Paris for the longest period is Charles de Gaulle. He was born in 1890 in Paris and spent much of his life in the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with new and exciting developments on the horizon. Here are some potential future trends in AI:
    
    1. Increased integration with other technologies: AI will continue to be integrated with other technologies, such as blockchain, sensors, and Internet of Things (IoT) devices. This will lead to a more seamless and interconnected world, where AI will work alongside other technologies to solve complex problems.
    
    2. AI in healthcare: AI will play a more significant role in healthcare, especially in diagnosing and treating diseases. With the help of AI-powered diagnostic tools, doctors and healthcare professionals will be able to detect diseases and diagnose patients with greater accuracy and speed


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

    name

    ]

     and

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     [

    person

    ality

    ].

     I

    'm

     currently

     [

    current

     location

    ]

     and

     I

     have

     [

    number

     of

     friends

    ].

     I

    'm

     a

     [

    character

    istics

     of

     the

     character

    ].

     And

     I

     enjoy

     [

    how

     I

     like

     to

     spend

     my

     free

     time

    ].

     What

     kind

     of

     character

     are

     you

    ?


    Hello

    ,

     my

     name

     is

     [

    name

    ]

     and

     I

     am

     a

     [

    age

    ]

     year

     old

     [

    gender

    ]

     [

    person

    ality

    ].

     I

    'm

     currently

     [

    current

     location

    ]

     and

     I

     have

     [

    number

     of

     friends

    ].

     I

    'm

     a

     [

    character

    istics

     of

     the

     character

    ].

     And

     I

     enjoy

     [

    how

     I

     like

     to

     spend

     my

     free

     time

    ].

     What

     kind

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     that

     hosts

     the

     E

    iff

    el

     Tower

    ,

     a

     landmark

     of

     the

     world

    .


    The

     answer

     is

    :
    


    Paris

     is

     the

     capital

     of

     France

     and

     hosts

     the

     E

    iff

    el

     Tower

    .

     Its

     rich

     history

    ,

     exquisite

     architecture

    ,

     and

     cultural

     landmarks

     make

     it

     a

     must

    -

    visit

     city

     for

     anyone

     visiting

     the

     country

    .

     The

     E

    iff

    el

     Tower

    ,

     located

     on

     the

     Champ

     de

     Mars

     neighborhood

    ,

     stands

     as

     a

     symbol

     of

     France

    's

     rich

     history

     and

     cultural

     heritage

    .

     Visitors

     can

     explore

     the

     world

    's

     tallest

     structure

    ,

     experience

     the

     iconic

     Notre

    -D

    ame

     Cathedral

    ,

     and

     admire

     the

     Notre

    -D

    ame

     de

     Paris

     Basil

    ica

    .

     Other

     attractions

     include

     the

     Lou

    vre

     Museum

    ,

     the

     Lou

    vre

     Pyramid

    ,

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     several

     trends

    :
    


    1

    .

     Increased

     integration

     with

     human

     AI

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     AI

     systems

    ,

     such

     as

     chat

    bots

    ,

     voice

     assistants

    ,

     and

     virtual

     assistants

    ,

     improving

     efficiency

     and

     personal

    ization

    .
    


    2

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     likely

     to

     become

     more

     widespread

    ,

     with

     self

    -driving

     cars

     becoming

     more

     common

     and

     safer

    .

     They

     will

     be

     able

     to

     navigate

     roads

    ,

     handle

     traffic

    ,

     and

     provide

     information

     to

     drivers

    .
    


    3

    .

     Chat

    bots

     and

     virtual

     assistants

    :

     AI

     chat

    bots

     and

     virtual

     assistants

     will

     become

     more

     sophisticated

    ,

     enabling

     them

     to

     provide

     more

     personalized

     and

     helpful

     responses

     to

     users

    .

     They

     will

     also

     be

     able

     to

     learn

     and

     improve

     over

    



```python
llm.shutdown()
```
