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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.61it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]

    Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.66it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=256):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=240):  48%|████▊     | 28/58 [00:04<00:02, 14.38it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]

    Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:04<00:00, 22.78it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 32.14it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.22 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   3%|▎         | 2/58 [00:00<00:02, 19.21it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.21 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.20 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):   9%|▊         | 5/58 [00:00<00:02, 22.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.10it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.18 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.16 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.14 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=960 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  31%|███       | 18/58 [00:00<00:01, 36.81it/s]Capturing num tokens (num_tokens=896 avail_mem=72.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=832 avail_mem=72.15 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=768 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=704 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=640 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  40%|███▉      | 23/58 [00:00<00:00, 40.24it/s]Capturing num tokens (num_tokens=576 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=512 avail_mem=72.12 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=480 avail_mem=72.14 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=448 avail_mem=72.12 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=416 avail_mem=72.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.11 GB):  48%|████▊     | 28/58 [00:00<00:00, 42.67it/s]Capturing num tokens (num_tokens=384 avail_mem=72.11 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=352 avail_mem=72.11 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=320 avail_mem=72.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=288 avail_mem=72.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=256 avail_mem=72.10 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=240 avail_mem=72.09 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.87it/s]Capturing num tokens (num_tokens=240 avail_mem=72.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=224 avail_mem=72.09 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=208 avail_mem=72.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]

    Capturing num tokens (num_tokens=192 avail_mem=72.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=176 avail_mem=72.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=160 avail_mem=72.08 GB):  66%|██████▌   | 38/58 [00:01<00:00, 39.79it/s]Capturing num tokens (num_tokens=160 avail_mem=72.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=144 avail_mem=72.08 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=128 avail_mem=72.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=112 avail_mem=72.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.32it/s]Capturing num tokens (num_tokens=96 avail_mem=72.07 GB):  74%|███████▍  | 43/58 [00:01<00:00, 39.32it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=72.07 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 32.89it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.36it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.23it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.78it/s]


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
    Generated text:  Ali Jafari, an Iraqi software developer, software engineer, and author from Baghdad. My professional achievements are in the fields of computer science, software development, and technology. My research interests are in artificial intelligence, computer vision, and machine learning. My personal interests are in technology, coding, and programming. I am passionate about building computers and the internet, and I am determined to contribute to the development of the technology. Here are some of the most interesting things that I have been working on in the field of artificial intelligence and computer vision:
    
    1. Deep learning research: My research in computer vision focuses on the development of deep learning models,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to finalize a plan that would bring in more money for the government. He has eight different items to choose from. Each item costs 7.5 dollars. How much would the president spend if he chooses the item that costs 7.5 dollars? The president would spend 7.5 dollars if he chooses the item that costs 7.5 dollars. 
    
    However, if we consider the context of the question, it seems like there might be a typo or an error in the question. If the president is trying to finalize a plan for the government, it's more likely that he's looking for a specific goal or plan
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of the country.
    
    Does it follow that if "The capital of France is the capital of the country" then "The country has the capital in France."?
    Choices:
    1). yes.
    2). it is not possible to tell.
    3). no.
    1).
    ===============================
    Prompt: The future of AI is
    Generated text:  more advanced, faster, and more personalized than ever before. From personalized marketing campaigns and personalized self-driving cars to artificial intelligence that can predict and prevent diseases, the potential of AI is vast and exciting. The question is how to harness and use this potential in a way that is ethical and responsible. AI is a powerful tool, but it can also be used in harmful ways. In this blog post, we will explore how AI is being used for both good and bad purposes, as well as how to ensure responsible use of this technology. You will also learn about some of the ethical questions surrounding AI, such as the potential for AI to create


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] and [Country]. I have [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] and [Country]. I have [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] and [Country]. I have [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] and [Country]. I have [Number] years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the Louvre Museum. It is also the birthplace of French literature and the home of the French Revolution. Paris is a bustling metropolis with a rich cultural heritage and a diverse population. The city is known for its fashion, art, and cuisine, and is a popular tourist destination. It is home to many famous landmarks and attractions, including the Notre-Dame Cathedral, the Louvre Museum, and the Champs-Élysées. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential future trends include:
    
    1. Increased integration with other technologies: AI is already being integrated into a wide range of devices and systems, but there is potential for even greater integration with other technologies such as sensors, blockchain, and quantum computing.
    
    2. Improved privacy and security: As AI systems become more complex and sophisticated, there will be increased concerns about privacy and security. There will be efforts to develop more secure and transparent AI systems that can be trusted to operate safely and ethically.
    
    3. Greater focus on ethical considerations:
    


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
    Generated text:  [Name] and I'm a/an [Occupation]. I've always been an [active/insistent/silent] type person. I'm a/an [Average height/Short, Medium height/ Tall, or Average weight/Very heavy/Very light] and [My favorite food], [My favorite hobby], and [My most annoying personality trait]. I enjoy [My hobbies], [My passions], and [My most annoying thing about being me]. I'm [Age], [Gender], and [Nationality], and I'm [My experience at the most recent work job]. [Insert a brief note or comment about me here
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, and it is located in the south of the country. The city is known for its rich history, culture, and architecture, which have made it a popular tourist destination. Paris has a diverse population of over 1 million people, and it is home to many famous landmarks and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also home to many iconic restaurants, cafes, and shops, making it a popular place to visit and shopping. Overall, Paris is a city of many charms and attractions that offers a unique experience for visitors.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, with potential trends and developments in the industry. Here are some of the most notable trends and developments that are expected to shape the future of AI:
    
    1. Increased Integration: With the advancement of artificial intelligence, it is expected that more sensors, machines, and devices will be integrated into the network to increase their intelligence and adaptability. This integration will enable them to process and analyze vast amounts of data more effectively, leading to better predictions, better decision-making, and more efficient operations.
    
    2. AI for Health: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve medical research. As AI becomes


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

     [

    Age

    ].

     I

     am

     [

    Occup

    ation

    ],

     [

    Job

     Title

    ].

     I

     am

     a

     [

    Length

     of

     Service

    ]

     year

     veteran

     in

     the

     [

    Industry

    /

    Field

    ]

     industry

    .

     I

     have

     [

    Number

     of

     significant

     achievements

    ]

     significant

     accomplishments

     in

     the

     [

    Industry

    /

    Field

    ].

     I

     am

     [

    Level

     of

     Service

    /

    Level

     of

     Experience

    ]

     level

     of

     experience

     in

     the

     [

    Industry

    /

    Field

    ].

     I

     have

     [

    Number

     of

     Languages

    /

    Fl

    u

    encies

    ]

     languages

    /

    flu

    encies

     and

     speak

     [

    Language

    ]

     flu

    ently

    .

     I

     am

     [

    Shape

     of

     Body

    ]

     in

     height

     and

     weight

    .

     I

     have

     [

    Number

     of

     Professional

     Cert

    ifications

    ]

     professional

     certifications

    .

     I

     have

     [

    Number

     of

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     love

    .

     Here

     are

     some

     additional

     facts

    :
    


    1

    .

     It

     is

     home

     to

     the

     E

    iff

    el

     Tower

    ,

     which

     has

     been

     the

     city

    's

     iconic

     symbol

     since

     

    1

    8

    8

    9

    .


    2

    .

     Paris

     is

     also

     famous

     for

     its

     annual

     Spring

     Festival

    ,

     known

     as

     the

     Festival

     de

     la

     Flor

    iss

    ance

    ,

     which

     takes

     place

     during

     the

     month

     of

     May

    .


    3

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     including

     the

     Lou

    vre

    ,

     which

     houses

     the

     world

    's

     largest

     collection

     of

     art

     and

     artifacts

    .


    4

    .

     Paris

     is

     also

     known

     for

     its

     rich

     culinary

     traditions

    ,

     with

     dishes

     like

     co

    q

     au

     vin

    ,

     esc

    arg

    ot

    ,

     and

     Bour

    gu

    ignon

     stew

     being

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     rapidly

     evolving

    ,

     with

     various

     trends

     expected

     to

     shape

     its

     development

     and

     impact

     in

     the

     coming

     years

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Automation

     and

     robotics

    :

     The

     integration

     of

     automation

     and

     robotics

     in

     various

     industries

     is

     expected

     to

     further

     improve

     efficiency

     and

     productivity

    .

     AI

    -powered

     robots

     and

     autonomous

     vehicles

     will

     become

     more

     common

    ,

     autom

    ating

     mundane

     tasks

     and

     freeing

     up

     human

     workers

     to

     focus

     on

     higher

    -value

     activities

    .
    


    2

    .

     Natural

     language

     processing

    :

     Advances

     in

     natural

     language

     processing

     will

     allow

     machines

     to

     understand

     human

     language

     more

     effectively

    ,

     enabling

     more

     natural

     interactions

     between

     humans

     and

     machines

    .

     This

     will

     create

     new

     possibilities

     for

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     self

    -driving

     cars

    .
    


    3

    



```python
llm.shutdown()
```
