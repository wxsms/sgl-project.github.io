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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.16it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.82it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.82it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.82it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.53it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:05<00:00, 21.51it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 30.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.73it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:07,  7.30it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.30it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:07,  7.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   5%|▌         | 3/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:05,  9.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   5%|▌         | 3/58 [00:00<00:05,  9.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:04, 11.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  19%|█▉        | 11/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.96it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:00<00:02, 19.96it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.22it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.22it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  29%|██▉       | 17/58 [00:01<00:01, 22.22it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.54it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.54it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.54it/s]Capturing num tokens (num_tokens=768 avail_mem=74.39 GB):  38%|███▊      | 22/58 [00:01<00:01, 28.54it/s]Capturing num tokens (num_tokens=768 avail_mem=74.39 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=704 avail_mem=74.05 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=640 avail_mem=74.30 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.13it/s]

    Capturing num tokens (num_tokens=576 avail_mem=74.06 GB):  43%|████▎     | 25/58 [00:01<00:01, 26.13it/s]Capturing num tokens (num_tokens=576 avail_mem=74.06 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.74it/s]Capturing num tokens (num_tokens=512 avail_mem=74.06 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.74it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.74it/s]Capturing num tokens (num_tokens=448 avail_mem=74.10 GB):  48%|████▊     | 28/58 [00:01<00:01, 23.74it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.10 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=416 avail_mem=74.11 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=384 avail_mem=74.11 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=352 avail_mem=74.14 GB):  53%|█████▎    | 31/58 [00:01<00:01, 22.66it/s]Capturing num tokens (num_tokens=352 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=320 avail_mem=74.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=288 avail_mem=74.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.64it/s]Capturing num tokens (num_tokens=256 avail_mem=74.25 GB):  59%|█████▊    | 34/58 [00:01<00:01, 23.64it/s]

    Capturing num tokens (num_tokens=256 avail_mem=74.25 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.03it/s]Capturing num tokens (num_tokens=240 avail_mem=74.24 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.03it/s]Capturing num tokens (num_tokens=224 avail_mem=74.23 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.03it/s]Capturing num tokens (num_tokens=208 avail_mem=74.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.03it/s]Capturing num tokens (num_tokens=192 avail_mem=74.22 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.03it/s]Capturing num tokens (num_tokens=192 avail_mem=74.22 GB):  71%|███████   | 41/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=176 avail_mem=74.21 GB):  71%|███████   | 41/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=160 avail_mem=74.21 GB):  71%|███████   | 41/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=144 avail_mem=74.20 GB):  71%|███████   | 41/58 [00:01<00:00, 27.65it/s]Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  71%|███████   | 41/58 [00:01<00:00, 27.65it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.16 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.86it/s]Capturing num tokens (num_tokens=112 avail_mem=74.17 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.86it/s]Capturing num tokens (num_tokens=96 avail_mem=74.17 GB):  78%|███████▊  | 45/58 [00:01<00:00, 29.86it/s] Capturing num tokens (num_tokens=80 avail_mem=74.18 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.86it/s]Capturing num tokens (num_tokens=64 avail_mem=74.17 GB):  78%|███████▊  | 45/58 [00:02<00:00, 29.86it/s]Capturing num tokens (num_tokens=64 avail_mem=74.17 GB):  84%|████████▍ | 49/58 [00:02<00:00, 31.79it/s]Capturing num tokens (num_tokens=48 avail_mem=74.16 GB):  84%|████████▍ | 49/58 [00:02<00:00, 31.79it/s]Capturing num tokens (num_tokens=32 avail_mem=74.16 GB):  84%|████████▍ | 49/58 [00:02<00:00, 31.79it/s]Capturing num tokens (num_tokens=28 avail_mem=74.14 GB):  84%|████████▍ | 49/58 [00:02<00:00, 31.79it/s]Capturing num tokens (num_tokens=24 avail_mem=74.15 GB):  84%|████████▍ | 49/58 [00:02<00:00, 31.79it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.15 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s]Capturing num tokens (num_tokens=20 avail_mem=74.15 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s]Capturing num tokens (num_tokens=16 avail_mem=74.14 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s]Capturing num tokens (num_tokens=12 avail_mem=74.13 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s]Capturing num tokens (num_tokens=8 avail_mem=74.13 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s] Capturing num tokens (num_tokens=4 avail_mem=74.12 GB):  91%|█████████▏| 53/58 [00:02<00:00, 33.33it/s]Capturing num tokens (num_tokens=4 avail_mem=74.12 GB): 100%|██████████| 58/58 [00:02<00:00, 35.63it/s]Capturing num tokens (num_tokens=4 avail_mem=74.12 GB): 100%|██████████| 58/58 [00:02<00:00, 25.19it/s]


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
    Generated text:  Louis Kassib. I have been married to the beautiful Svetlana Solovyeva for 21 years, and we have a daughter, Stefka. We are very supportive of the LGBTQ+ community and we enjoy traveling. What are some of your favorite memories from your time with Svetlana? As an AI language model, I don't have personal experiences, but I can tell you some of Louis Kassib's favorite memories with Svetlana Solovyeva.
    
    One of Louis and Svetlana's most memorable times together was when they were trying to travel to a remote, challenging location in the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person. (1) They hold a high office. (2) It is a ceremonial position. (3) They are not elected. (4) They do not represent the people. (5) In this role, they are often in charge.
    The president is a person, a person, a person, a person, a person. So, the answer is a person. The president is a person because it is a person who holds a high office in the government. The office is ceremonial and not elected. The position does not represent the people, but it is often in charge. The president is a person, therefore,
    ===============================
    Prompt: The capital of France is
    Generated text:  the capital of France. 
    
    Does this mean that the capital of France is located in a city? 
    
    Select from the following. * no; * yes;
    
    **no** The capital of France is located in a city, not a country. It is the main city where the government, parliament, and most important institutions are based. While cities are important places within the country, the capital is a specific location and city. So the answer is "no". Here's a summary of the reasoning:
    
    1. The capital is not a country but a city.
    2. The capital is defined as the main city where the government, parliament,
    ===============================
    Prompt: The future of AI is
    Generated text:  becoming increasingly uncertain, and understanding the potential implications of the changing landscape of artificial intelligence is essential to inform effective strategies for addressing its challenges and leveraging its opportunities.
    
    The future of AI is not just about developing new algorithms and technologies. It's also about understanding the ethical implications of AI, and how we can use it to create a more just and equitable world. This requires a deep dive into the history and history of AI, including the early developments and the challenges faced in developing and implementing AI systems.
    
    As we continue to push the boundaries of AI, we should also be aware of the potential risks and unintended consequences of AI. This includes concerns about


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and what you're looking for in a job. Let's chat! [Name] [Company Name] [Company Address] [City, State, Zip Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and is home to many world-renowned museums, art galleries, and restaurants. It is also a major center for business and finance, with many of the world's largest companies headquartered there. The city is known for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we can expect to see even more widespread use
    


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
    Generated text:  [Your Name], and I am a professional [your position or profession]. I have been working with [your company name] for [number of years] years, and I am always looking for opportunities to grow my skills and learn new things. I'm confident in my ability to adapt to different environments and work well in a team, and I'm eager to make a difference in the world. I'm always looking for new challenges and opportunities to grow my career. What's your favorite hobby or activity to do? My favorite hobby is hiking and spending time outdoors. How do you stay motivated and focused on your goals? I set achievable goals
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city and the third most populous city in the European Union.
    
    To solve the above logical reasoning problem, follow these steps:
    
    1. Carefully read and comprehend the provided text.
    2. Identify the key elements in the sentence:
       - Paris is the capital of France
       - It is the largest city in France
       - It is the third most populous city in the European Union
    3. Formulate a concise factual statement using these key elements.
    4. Revise the statement to ensure it is clear, concise, and error-free.
    
    Following these steps, the factual statement about France's capital city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be driven by several trends, including:
    
    1. Increased integration of AI into various industries: As AI becomes more widely integrated into various sectors, from healthcare to finance to manufacturing, we can expect to see more complex and sophisticated AI systems that can solve complex problems more effectively than human AI.
    
    2. More personalized and automated customer experiences: AI will continue to become more sophisticated, with the ability to personalize customer experiences, automate routine tasks, and provide more seamless and convenient interactions with digital products and services.
    
    3. Greater emphasis on ethical and responsible AI: As concerns about AI's impact on society and the environment grow, there will be an


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

    ],

     and

     I

    'm

     a

     [

    type

    ]

     [

    gender

    ]

     [

    age

    ]

     [

    location

    ]

     from

     [

    country

    ].

     I

     am

     a

     [

    occupation

    ],

     [

    job

     title

    ],

     and

     I

     enjoy

     [

    activities

    ].

     I

    'm

     passionate

     about

     [

    my

     profession

    ],

     and

     I

    'm

     constantly

     learning

     and

     growing

     in

     my

     field

    .

     I

    'm

     also

     known

     for

     [

    my

     personal

     traits

    ],

     and

     I

    'm

     a

     great

     listener

    .

     I

     believe

     in

     the

     power

     of

     [

    mot

    iv

    ational

     quote

    ],

     and

     I

    'm

     committed

     to

     [

    my

     cause

     or

     goal

    ].

     How

     do

     you

     like

     to

     spend

     your

     free

     time

    ?

     My

     favorite

     activities

     include

     [

    list

     of

     activities

    ].


    I

     hope

     you

     enjoy

     meeting

     you

    !

     Let

     me

     know

     if

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     is

     the

     largest

     city

     in

     Europe

     by

     population

    .

     Its

     architecture

    ,

     cuisine

    ,

     and

     cultural

     heritage

     are

     renowned

     worldwide

    .

     Paris

     is

     known

     for

     its

     op

    ulent

     bou

    lev

    ards

    ,

     charming

     museums

    ,

     and

     iconic

     landmarks

     like

     the

     E

    iff

    el

     Tower

    .

     It

    's

     a

     major

     transportation

     hub

     for

     Europe

     and

     holds

     significant

     cultural

     and

     political

     importance

    .

     France

     is

     a

     country

     with

     a

     rich

     history

     and

     diverse

     culture

    ,

     and

     Paris

     is

     a

     perfect

     blend

     of

     old

    -world

     charm

     and

     modern

     sophistication

    .

     The

     city

     attracts

     millions

     of

     tourists

     each

     year

     due

     to

     its

     architecture

    ,

     fashion

    ,

     cuisine

    ,

     and

     sense

     of

     French

    ness

    .

     Paris

     is

     also

     an

     important

     center

     for

     business

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     see

     continued

     growth

     and

     development

     in

     several

     key

     areas

    .

     One

     potential

     trend

     is

     the

     integration

     of

     AI

     into

     everyday

     life

    ,

     with

     more

     widespread

     adoption

     of

     AI

     in

     transportation

    ,

     healthcare

    ,

     and

     consumer

     products

    .

     Additionally

    ,

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     the

     internet

     of

     things

     (

    Io

    T

    )

     and

     the

     cloud

    ,

     as

     these

     technologies

     become

     more

     prevalent

     in

     everyday

     life

    .
    


    Another

     potential

     trend

     is

     the

     expansion

     of

     AI

     into

     the

     business

     sector

    ,

     with

     more

     businesses

     and

     organizations

     adopting

     AI

     for

     tasks

     such

     as

     data

     analysis

    ,

     customer

     service

    ,

     and

     predictive

     maintenance

    .

     AI

    -powered

     chat

    bots

     and

     virtual

     assistants

     are

     also

     expected

     to

     become

     more

     widely

     available

    ,

     offering

     improved

    



```python
llm.shutdown()
```
