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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:51,  4.06s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.59it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.73it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.73it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.54it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.82it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.14it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.15it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.60it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.75it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.77it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.06it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 44.06it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.06it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 44.06it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 44.89it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.17it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.54it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 38.85it/s]


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
    Generated text:  Fenny and I'm from Sydney. I'm 16 years old and I have a pretty healthy diet. I'm a competitive athlete. I'm also the captain of the Sydney Knights, a soccer team. I'm on the team for five years. I'm in the soccer team because I'm very good at it. I play soccer every day. I'm in the girls' team. I'm in the first team. It's the best team I've played for. I'll play for the team for ever. I love to run. I'm in the girls' team. I play soccer every day. I run every
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official who serves as the head of state and the commander-in-chief of the armed forces. This office was created in 1787. However, the current U.S. president is not the first elected president. The first president of the United States was George Washington. He was elected in 1788. The first president of the United States was George Washington.
    This article will discuss the United States presidential election process and the election results. This process is called the Electoral College. The Electoral College is one of the two chambers of the United States Congress. The remaining chamber is the U.S. Senate. This article
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most visited city in the world. It is the capital of France and the largest city in metropolitan France. It is the second largest city in the United Kingdom and the third largest city in the European Union. With over 8 million inhabitants, Paris is the largest metropolitan area in the world. It's the third largest metropolis in the world. It has the most livable city in the world. Paris is the capital of France, and the largest city in metropolitan France. It's the largest city in metropolitan France. Paris is the capital of France, and the largest city in metropolitan France. It is the second
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be a bright one. While we should all be excited about the possibilities, it's also important to keep in mind that AI is not yet omnipotent and will always be a tool that we use to solve problems. It's crucial to be mindful of the ethical considerations associated with the development and use of AI, and to ensure that we are working towards a world where AI is used responsibly and ethically.
    AI is already used in many industries, including healthcare, finance, and transportation. It's not yet perfect, but it's a valuable tool that can help us make better decisions and solve problems more efficiently.
    AI is also used


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and I'm always looking for ways to [job title]. I'm always eager to learn and grow, and I'm always looking for opportunities to contribute to the company. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on art and literature. It is also a popular tourist destination, with millions of visitors annually. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Its reputation as a city of art, culture, and sophistication is well-documented. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for the development and deployment of AI systems.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to a more human-like experience with AI, but also raise
    


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
    Generated text:  [Your Name], and I'm a [Your Position/Title]. I've always been passionate about [Your Passion/Interest], and I'm dedicated to [Your Mission/Goal]. I believe that success is not just about achieving your goals, but also about making a difference in the lives of those around you. I'm always looking for new ways to inspire and educate others, and I'm always eager to learn from others and grow as a person. I'm open-minded, adaptable, and always eager to learn new things. I'm excited to be a part of [Your Team/Company] and contribute to its success. Thank you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. It is the cultural and economic hub of the country and a major international center for art, fashion, and cuisine. Paris has been an influential global city for over 300 years and continues to be a key part of France’s cultural and political identity. Its status as the capital is made more prestigious by its status as a European city, with many of the country’s most important universities, government offices, and offices of the European Union headquarters.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a rapid evolution and diversification of the technology and its applications. Some possible trends that could emerge in the near future include:
    
    1. More advanced and distributed AI: AI models that are able to learn and adapt to new data in real-time, with the capability to learn and evolve over time, could become more prevalent.
    
    2. Personalized and context-aware AI: As data on individuals becomes more accessible and available, AI models will be able to learn more about a person's behavior, preferences, and context, leading to more personalized and context-aware AI applications.
    
    3. Autonomous and adaptive AI: As autonomous vehicles,


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

     I

     am

     a

     [

    Occup

    ation

    ]

     with

     [

    Years

     of

     Experience

    ],

     and

     I

     have

     a

     passion

     for

     [

    Your

     Hobby

     or

     Interest

    ].

     I

     love

     to

     [

    Describe

     Your

     Hobby

     or

     Interest

    ].

     Whether

     it

    's

     [

    What

     the

     Hobby

     or

     Interest

     is

    ],

     I

    'm

     always

     looking

     for

     ways

     to

     [

    What

     I

    'm

     Doing

     to

     Keep

     Me

     Mot

    ivated

     or

     Happy

    ].

     If

     you

    'd

     like

     to

     chat

    ,

     please

     feel

     free

     to

     reach

     out

     and

     let

    's

     get

     to

     know

     each

     other

    !

     [

    Start

     your

     short

     self

    -int

    roduction

    ,

     avoiding

     any

     personal

     or

     emotional

     elements

    ]

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     I

     am

     a

     [

    Occup

    ation

    ]

     with

     [

    Years

     of

     Experience

    ],

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

    .

     True

     B

    .

     False

    .

     A

    .

     True

    .

     Paris

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     a

     major

     metropolitan

     area

     and

     a

     leading

     international

     city

    .

     The

     city

     is

     home

     to

     the

     government

    ,

     parliament

    ,

     government

     ministries

    ,

     and

     administrative

     agencies

    .

     It

     is

     also

     a

     major

     transport

     hub

    ,

     with

     a

     well

    -develop

    ed

     metro

     system

    ,

     which

     connects

     the

     city

     to

     many

     other

     major

     cities

     in

     Europe

     and

     beyond

    .

     Paris

     has

     a

     rich

     cultural

     heritage

     and

     is

     a

     major

     center

     for

     art

    ,

     design

    ,

     architecture

    ,

     and

     music

    .

     It

     is

     also

     a

     cosm

    opolitan

     city

     with

     a

     diverse

     population

     of

     around

     

    2

     million

     inhabitants

    .

     Paris

     is

     an

     important

     tourist

     destination

     and

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

     and

     can

     be

     difficult

     to

     predict

    ,

     but

     there

     are

     many

     potential

     trends

     that

     could

     shape

     the

     technology

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     trends

     that

     could

     be

     present

     in

     the

     AI

     field

     in

     the

     next

     decade

    :
    


    1

    .

     AI

     will

     continue

     to

     be

     integrated

     into

     everyday

     life

    ,

     from

     the

     devices

     we

     use

     to

     the

     services

     we

     access

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     AI

     technologies

     that

     are

     even

     more

     integrated

     into

     our

     lives

    ,

     such

     as

     voice

     recognition

    ,

     self

    -driving

     cars

    ,

     and

     chat

    bots

    .
    


    2

    .

     AI

     will

     continue

     to

     evolve

    ,

     with

     new

     types

     of

     algorithms

     and

     neural

     networks

     emerging

    .

     This

     could

     lead

     to

     the

     development

     of

     new

     applications

     and

     services

     that

     are

    



```python
llm.shutdown()
```
