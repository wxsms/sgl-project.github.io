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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.69it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:07,  4.35s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.49it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.45it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.45it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.97it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.02it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.24it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.29it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.29it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.29it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.29it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.29it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 22.64it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  55%|█████▌    | 32/58 [00:00<00:00, 45.83it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.24it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.24it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.24it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:00<00:00, 47.24it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  66%|██████▌   | 38/58 [00:01<00:00, 47.24it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  76%|███████▌  | 44/58 [00:01<00:00, 48.41it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 46.80it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.29it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.29it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  93%|█████████▎| 54/58 [00:01<00:00, 46.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 42.63it/s]


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
    Generated text:  Hiroshi. I have been a lifelong college student and have had a lot of experience with the life in college. I currently work at a computer company. At work, I have to learn a lot of new things, and have to put in a lot of time to do so. And of course, my job is not only to work, but also to learn, which makes it both exhausting and exciting for me. However, there are a few things that I wish I did more. First, I wish I could get more sleep. I get up and go to work, and I sleep in. I'm not sure why, but
    ===============================
    Prompt: The president of the United States is
    Generated text:  a part of the executive branch of the government. The president has many jobs, including the responsibility for making and signing treaties. The president's duties include making and signing executive orders. The president also has a role in the defense of the country. But, the president does not have the power to make laws and also does not have the power to make treaties.
    It is the responsibility of the congress to make and sign legislation that must be passed by the president. The president does not have the power to sign legislation. The president's power to veto legislation passed by the congress does not affect the president's power to sign it.
    Can we infer
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The capital of Russia is Moscow. The capital of Britain is London. In which country is the country of the capital of the country of the capital of the country of the capital of the world's capital not located? Let's program in Python in the response.
    
    def find_country():
        countries = ["France", "Russia", "Britain", "London"]
        return countries.index(max(countries))
    
    print(find_country())  # Output: "Britain"
    In the given Python program, the function find_country() is used to determine the capital of the country of the capital of the country of the capital of the world's capital. This
    ===============================
    Prompt: The future of AI is
    Generated text:  digitalization
    
    The future of AI is digitalization
    
    The future of AI is digitalization
    
    One of the most exciting things that happens in AI is the digitization of the data. It's a process that has taken place for many decades, but it's still the future of AI. There are some key ways in which this digitization is taking place, and I'm going to talk about some of them in this article.
    
    AI in Digitalization
    
    AI in Digitalization
    
    The central idea of digitalization is to use digital technology to store and analyze data. This can be anything from customer data to medical records to social media data.


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason for interest in the industry] and I'm always eager to learn and grow. I'm a [reason
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to many international organizations and institutions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is a symbol of France and a major hub of global affairs. Paris is a city of love, passion, and creativity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation and robotics: AI is already being used in manufacturing, healthcare, and transportation, and we can expect to see even more automation and robotics in the future. This will lead to increased efficiency, cost savings, and job displacement.
    
    2. Enhanced cognitive abilities: AI is already capable of learning and improving its performance, and we can expect to see even more advanced cognitive abilities in the future. This will lead to more accurate predictions, better decision-making, and even self-awareness.
    
    3. Improved privacy and security: As AI becomes more integrated into our daily
    


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
    Generated text:  [Name] and I'm a [age] year old, [gender] and [occupation] [Name]. I am enthusiastic about [a hobby or interest, such as music, cooking, or dancing] and I strive to make the world a better place by sharing my knowledge and experience with others. I am always willing to learn and grow as a person, and I am excited to share my knowledge with everyone who wants to learn.
    
    **Note:** Feel free to add or remove any details as you see fit, as long as they maintain the essence of your self-introduction. Let me know if you'd like me to revise anything
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located in the region of Occitanie in the south of the country. Paris is the largest city and the second most populous city in the European Union, and the third most populous in the world. It is also the capital of the department of Paris, which has a population of 1.4 million. Paris is famous for its cuisine, art, and fashion, and is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its vibrant culture and history, and is a major cultural center in France. It is a global city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a multitude of factors, including advances in computing power, advances in data analysis, advances in machine learning, and advances in self-driving technology. Here are some possible trends that could shape the future of AI:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into daily life, there is a growing awareness of its potential impact on society. There is a growing emphasis on ethical considerations, including privacy, bias, and accountability.
    
    2. Automation of tasks: As AI becomes more powerful and capable, it is likely to be used more widely to automate tasks, reducing the need for human workers. This could lead


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

     a

     passionate

     and

     dedicated

     freelance

     writer

     who

     specializes

     in

     creating

     engaging

     and

     creative

     writing

    .

     I

    've

     written

     for

     print

    ,

     radio

    ,

     and

     online

     publications

     and

     have

     a

     love

     for

     crafting

     stories

     that

     inspire

     and

     capt

    ivate

     readers

    .

     I

     believe

     in

     the

     power

     of

     words

     to

     move

     people

     and

     leave

     them

     with

     a

     sense

     of

     wonder

     and

     adventure

    .

     Whether

     it

    's

     writing

     poetry

    ,

     short

     stories

    ,

     or

     novel

     fiction

    ,

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     improve

     as

     a

     writer

    .

     I

    'm

     excited

     to

     be

     a

     part

     of

     this

     journey

     and

     work

     with

     others

     who

     share

     my

     passion

     for

     creativity

     and

     storytelling

    .

     Welcome

     to

     [

    Your

     Name

    ],

     the

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

    .

     It

     is

     a

     historical

     and

     cultural

     center

     with

     landmarks

     such

     as

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Mar

    ne

     River

    .

     Paris

     is

     also

     known

     for

     its

     fashion

     industry

    ,

     gastr

    onomy

    ,

     and

     wine

     industry

    .

     The

     city

     has

     a

     diverse

     population

     and

     a

     vibrant

     street

     life

    .

     It

     is

     a

     major

     hub

     for

     tourism

     and

     business

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     With

     its

     grand

     architecture

    ,

     beautiful

     landscapes

    ,

     and

     rich

     culture

    ,

     Paris

     is

     a

     city

     of

     endless

     possibilities

    .

     (

    example

     sentence

    )

     Paris

     is

     a

     major

     city

     in

     France

    ,

     famous

     for

     its

     historical

     landmarks

    ,

     fashion

     industry

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     changing

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     emerge

     as

     the

     technology

     continues

     to

     evolve

    .

     Here

     are

     some

     potential

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

     focus

     on

     ethical

     considerations

    :

     As

     AI

     continues

     to

     develop

    ,

     it

     will

     become

     more

     important

     to

     consider

     the

     ethical

     implications

     of

     its

     use

    .

     This

     could

     lead

     to

     a

     greater

     focus

     on

     ethical

     considerations

     from

     the

     start

     of

     the

     development

     process

    ,

     as

     well

     as

     throughout

     the

     product

    's

     lifecycle

    .
    


    2

    .

     Improved

     transparency

     and

     accountability

    :

     AI

     systems

     are

     becoming

     more

     sophisticated

     and

     autonomous

    ,

     but

     there

     is

     still

     a

     lack

     of

     transparency

     and

     accountability

     in

     how

     they

     are

     used

    .

     To

     address

     this

    ,

     AI

     developers

     will

     need

     to

    



```python
llm.shutdown()
```
