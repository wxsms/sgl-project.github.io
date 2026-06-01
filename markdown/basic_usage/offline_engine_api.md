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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:13,  4.44s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.35it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.35it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.75it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.68it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.68it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:02, 18.69it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.16 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.15 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.15 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.14 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.13 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.13 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s]Capturing num tokens (num_tokens=960 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.12 GB):  31%|███       | 18/58 [00:00<00:01, 36.47it/s]Capturing num tokens (num_tokens=896 avail_mem=74.12 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=832 avail_mem=74.12 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=768 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=704 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=640 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=576 avail_mem=74.11 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.93it/s]Capturing num tokens (num_tokens=576 avail_mem=74.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=512 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=480 avail_mem=74.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=416 avail_mem=74.10 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=384 avail_mem=74.10 GB):  48%|████▊     | 28/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=384 avail_mem=74.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.55it/s]Capturing num tokens (num_tokens=352 avail_mem=74.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.55it/s]Capturing num tokens (num_tokens=320 avail_mem=74.09 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.55it/s]Capturing num tokens (num_tokens=288 avail_mem=74.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=256 avail_mem=74.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=240 avail_mem=74.08 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.55it/s]Capturing num tokens (num_tokens=240 avail_mem=74.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=224 avail_mem=74.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=208 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=176 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=160 avail_mem=74.07 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=160 avail_mem=74.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=144 avail_mem=74.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=128 avail_mem=74.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=112 avail_mem=74.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=96 avail_mem=74.06 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s] Capturing num tokens (num_tokens=80 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.77it/s]Capturing num tokens (num_tokens=80 avail_mem=74.05 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=64 avail_mem=74.05 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.67it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.50it/s]

    Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 37.62it/s]


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
    Generated text:  Michael and I'm a software engineer. I'm not a politician, I'm a scientist. I'm looking forward to getting together with you today. I am a scientist and not a politician. I will be discussing topics related to nanotechnology and advanced materials. Are you interested in learning about advanced materials and nanotechnology? If so, what is your area of interest? Additionally, I am not a politician, so I may have a different perspective on certain topics. I want to ensure that we have an open and respectful exchange of ideas. How can I help you today? We can discuss a variety of topics related to nanotechnology and advanced
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, but he also has some of the least popular jobs. He is the boss of the country, but people don't like him. The president also helps run the country, but he only gets a small part of the money. He is usually very busy, but he doesn't get to see his family very much. The president is like a king, but he is also like a mayor. His job is to make decisions and take care of the country. Sometimes the president has to use some of the power of the country that other people don't have. He has the power to tell people what to do and how
    ===============================
    Prompt: The capital of France is
    Generated text:  ____.
    A. Paris
    B. Lyon
    C. Marseille
    D. Nice
    Answer:
    
    A
    
    Which of the following is NOT a basic characteristic of a public institution?
    A. Non-profit nature
    B. Limited scale
    C. Strong external connection
    D. Unique qualification
    Answer:
    
    B
    
    The most common site of metastasis for invasive mole is
    A. Liver
    B. Kidney
    C. Lung
    D. Brain
    E. Spleen
    Answer:
    
    A
    
    According to the "Regulations on the Management of Drug Instructions and Labels", what should the drug name and indications be marked with?
    A
    ===============================
    Prompt: The future of AI is
    Generated text:  here; however, we are at a disadvantage due to the fact that we have not taught our brains how to think. That is why it is important to ensure that our brains have access to an interface that allows them to learn. This can be done with the use of neural networks.
    In the current era, we are seeing more and more reliance on artificial intelligence and machine learning. AI and machine learning is a fast-growing field with significant potential for transforming industries and solving complex problems.
    The future of AI is here, but it is important to ensure that the design of the interface to train our brains is appropriate and effective. This can be achieved


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [job title] at [company name]. I'm passionate about [job title], and I love [job title] because [reason for passion]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for new ways to explore and discover new things. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that hosts the Eiffel Tower and is known for its rich history and cultural heritage. It is also the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a vibrant and dynamic city with a rich cultural and historical heritage. It is a popular tourist destination and a major economic center in France. The city is also known for its cuisine, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is a growing emphasis on developing AI that is designed to be ethical and responsible. This could mean that AI systems are designed to minimize harm to individuals and society as a whole, and that they are transparent and accountable.
    
    2. AI will become more integrated with other technologies: As AI becomes more integrated with other technologies, such as sensors, cameras, and other forms of data collection, it is likely
    


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
    Generated text:  [Name]. I'm a [Position] at [Company]. I work here to [Job Description], and I love what I do. If you have any questions or would like to learn more about me, please feel free to reach out. Thank you! [Name] is [Name], a [Title] at [Company]. I am dedicated to [Job Description] and I enjoy [Role Responsibilities]. If you have any questions or would like to learn more about me, please feel free to reach out. Thank you! I'm a [Age], [Gender], [Race], [Disability Status] [Name] is [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Palace. The city is also renowned for its rich history, including the presence of the famous Louvre Museum, which houses an extensive collection of art and artifacts from throughout history. Paris is also famous for its fashion industry and fashion shows, as well as its climate and cuisine, which are all part of its vibrant and diverse culture. 
    
    (Note: I've chosen a general topic to avoid getting too specific about a particular city and to maintain a general tone.) 
    
    French capital city Paris is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and changing rapidly, with many possibilities and possibilities of future developments. Here are some possible future trends in AI:
    
    1. Deep learning: AI will continue to develop deep learning, which involves algorithms that can learn and improve without being explicitly programmed. This will enable machines to solve complex problems and perform tasks that would be impossible for humans to do.
    
    2. Ethics and governance: As AI technology becomes more widely adopted, there will be a need for clear guidelines and regulations to ensure that AI is used ethically and responsibly. This will require collaboration between governments, ethicists, and industry leaders to develop policies that protect the rights and privacy of


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

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    job

     title

    ]

     who

     has

     always

     been

     interested

     in

     [

    what

     you

     did

     or

     are

     doing

     that

     you

    're

     proud

     of

    ].

     I

    'm

     passionate

     about

     [

    what

     you

    're

     passionate

     about

    ],

     and

     I

     enjoy

     [

    how

     I

    'm

     passionate

     about

     it

    ].

     I

    'm

     [

    how

     you

     feel

     about

     yourself

    ,

     e

    .g

    .

     '

    person

    able

    ',

     '

    conf

    ident

    ',

     '

    creative

    ',

     '

    ener

    getic

    ']

     and

     I

     love

     [

    how

     you

     want

     others

     to

     perceive

     you

    ,

     e

    .g

    .

     '

    me

    ek

    ',

     '

    conf

    ident

    ',

     '

    strong

    ']

    .


    As

     a

     [

    job

     title

    ],

     I

    'm

     [

    how

     you

     feel

     about

     it

    ]

     and

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    While

     Paris

     is

     known

     as

     the

     city

     of

     love

    ,

     it

     has

     a

     diverse

     culture

     with

     influences

     from

     various

     ethnic

     groups

     and

     traditions

    .

     The

     city

     has

     a

     population

     of

     over

     one

     million

     and

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     has

     a

     strong

     economy

     and

     a

     vibrant

     nightlife

    ,

     and

     it

     is

     a

     popular

     tourist

     destination

     and

     business

     center

    .

     It

     is

     often

     called

     the

     "

    City

     of

     Light

    "

     and

     "

    The

     Flower

     City

    "

     due

     to

     its

     historic

     architecture

     and

     modern

     culture

    .

     With

     its

     rich

     history

    ,

     artistic

     heritage

    ,

     and

     cosm

    opolitan

     community

    ,

     Paris

     is

     a

     city

     that

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     rapid

     and

     ever

    -exp

    anding

     range

     of

     capabilities

    ,

     applications

    ,

     and

     technologies

    .

     Some

     of

     the

     key

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     AI

     ethics

     and

     fairness

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     there

     is

     growing

     interest

     in

     addressing

     issues

     of

     bias

    ,

     discrimination

    ,

     and

     fairness

    .

     This

     includes

     developing

     ethical

     guidelines

     and

     practices

     for

     developing

     and

     deploying

     AI

     systems

    ,

     as

     well

     as

     ensuring

     that

     AI

     systems

     do

     not

     perpet

    uate

     or

     exacerb

    ate

     existing

     social

     inequalities

    .
    


    2

    .

     Deep

     learning

     and

     multi

    -modal

     AI

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     there

     is

     a

     growing

     focus

     on

     developing

     deep

     learning

     and

     multi

    -modal

     AI

     approaches

     that

     can

     better

     understand

     and

    



```python
llm.shutdown()
```
