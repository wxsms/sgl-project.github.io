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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.22it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:48,  5.07s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.28it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.48it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.83it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:05<00:02, 12.65it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]

    Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 18.50it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=20):  78%|███████▊  | 45/58 [00:05<00:00, 26.20it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 34.81it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 34.81it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 34.81it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 34.81it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 34.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 16.06it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.80it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   7%|▋         | 4/58 [00:00<00:03, 16.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 16.80it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.65it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.65it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 19.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.35it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.61it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 36.18it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.46it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.46it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 36.46it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.46it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:01<00:00, 36.46it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.56it/s]

    Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 30.56it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 30.12it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.20it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.20it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.20it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.67it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.67it/s]

    Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.67it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 35.67it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 31.83it/s]


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
    Generated text:  Priscilla Lopez. I am a PhD student in the Department of Computational Science and Engineering at MIT and a researcher at the Center for Innovative Science and Technology. I’m interested in the intersection of machine learning and data science, with an emphasis on the use of deep learning models and predictive analytics to improve medical diagnoses and personalized medicine. My work is motivated by a curiosity to find ways to use data science to improve human health, and to explore the potential of deep learning for medical applications.
    My research focuses on using deep learning models for predicting the outcome of medical diagnoses. The first stage of my research, called “Learning to Rank,” involves finding
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is the boss of the country. He is supposed to be as good as possible to the people. Many people think he is very important. He is usually very busy all the time. Most of the time, people think he is not very friendly. Many people think he is not as intelligent as the other people. A lot of people think he is too serious. He has a very important job. The President is like a king or a queen. He is always wearing a red suit. He has a very important job. He is also the head of the executive branch. What is the President? The President
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Its history dates back to the XIIth century. During the French Revolutionary Wars, it was the capital of the National Constituent Assembly, which eventually led to the formation of the First Republic, or the First French Republic, and the start of the French Revolutionary Wars, which led to the formation of the Second Republic. After the fall of the Second Republic, the Directory took over the country until the formation of the Third Republic.
    Paris has a large population and is the largest city in France by area. It has a long history, and it has been influenced by many cultures and religions throughout the centuries. Paris is famous for its art
    ===============================
    Prompt: The future of AI is
    Generated text:  in motion, and its impact on various industries is causing a chain reaction of changes. The AI industry is undergoing a transformation in the domain of machine learning and deep learning, driven by the increasing use of quantum computers and the ongoing development of AI models based on large-scale data. As such, the industry is evolving to become more predictive, efficient, and adaptive to the evolving needs of the market.
    One of the key factors driving this evolution is the increasing role of AI in healthcare. AI-powered diagnostic tools are being developed to provide patients with accurate and fast results, leading to improved health outcomes. Additionally, AI is being used to monitor patient behavior


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is known for its cuisine, fashion, and art scene, and is a popular destination for international visitors. It is also home to many important historical sites and landmarks, including the Louvre and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a greater emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their work and strive to create systems that are fair, transparent, and accountable.
    
    2. Greater integration with other technologies: AI is likely to become more integrated with
    


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
    Generated text:  [Name]. I'm a [Dress] with [Description]. I'm a [Role]. And I love to [Favorite Activity].
    I'm [Age] years old. I'm [Occupation]. And I'm really passionate about [Favorite Hobby/Activity/Interest]. I look forward to [Importance] in this life, and I believe that [Purpose/Goal]. I’m a [Character] who wants to [Be] [role]. And I believe that I can be [role].
    I'm a [Name], a [Dress] with [Description], a [Role], and a [Favorite Activity]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower and the Louvre Museum. It also has a rich history and is a popular tourist destination, hosting numerous famous events such as the French New Year celebration. Paris is known for its art, culture, and cuisine, and is a major center of politics and commerce. It is also home to several museums and public spaces such as the Notre-Dame Cathedral and the Marais district, which are recognized as Paris's cultural heritage. Overall, Paris is an important and vibrant city with a unique character. - The capital of France, Paris is known for its iconic landmarks such as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid development, increased efficiency, and broader adoption across various industries. Some possible trends include:
    
    1. Increased integration with existing technologies: AI will continue to integrate more seamlessly with existing technologies such as the Internet of Things, machine learning, and natural language processing, making it easier for machines to learn and adapt to new environments.
    
    2. Expansion of AI to include more complex tasks: As AI becomes more adept at recognizing patterns and making decisions, it is likely to be applied to a wider range of tasks, such as medical diagnosis, autonomous vehicles, and more complex decision-making in industries like finance and manufacturing.
    
    3. AI


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

     __

    ________

    __.

     I

    'm

     an

     electric

    ian

     with

     a

     passion

     for

     science

     and

     technology

    .

     My

     work

     with

     electrical

     systems

     is

     invaluable

    ,

     and

     I

     enjoy

     creating

     solutions

     to

     problems

     that

     require

     precision

    ,

     efficiency

    ,

     and

     durability

    .

     I

     also

     love

     to

     travel

     and

     explore

     the

     world

    ,

     and

     I

     enjoy

     learning

     about

     new

     places

     and

     cultures

     through

     my

     travel

     writing

    .

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

     as

     an

     electric

    ian

     and

     a

     writer

    .

     Thank

     you

     for

     considering

     my

     self

    -int

    roduction

    .

     [

    insert

     the

     name

     of

     the

     fictional

     character

     you

     created

    ]

     Here

    's

     a

     neutral

     self

    -int

    roduction

     for

     a

     fictional

     character

    :


    I

    'm

     an

     electric

    ian

     with

     a

     passion

     for

     science

     and

     technology

    .

    
    
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

     known

     for

     its

     rich

     history

    ,

     charming

     architecture

    ,

     and

     vibrant

     culture

    .

     Its

     UNESCO

     World

     Heritage

     site

     includes

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

     also

     home

     to

     the

     headquarters

     of

     many

     major

     French

     companies

     and

     is

     a

     major

     center

     for

     art

    ,

     fashion

    ,

     and

     science

    .

     Paris

     is

     renowned

     for

     its

     cultural

     and

     entertainment

     offerings

    ,

     including

     the

     annual

     E

    iff

    el

     Tower

     Festival

    ,

     the

     Opera

     House

    ,

     and

     the

     annual

     Jazz

     Festival

    .

     It

     is

     also

     the

     birth

    place

     of

     many

     famous

     French

     artists

     and

     writers

    ,

     including

     Dante

     Al

    igh

    ieri

    ,

     Charles

     Dickens

    ,

     and

     Oscar

     Wilde

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     there

     are

     a

     number

     of

     trends

     that

     are

     likely

     to

     shape

     the

     technology

     and

     applications

     in

     the

     coming

     years

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     is

     able

     to

     learn

     from

     large

     datasets

     and

     provide

     more

     accurate

     and

     personalized

     health

     diagnoses

    ,

     the

     use

     of

     AI

     in

     healthcare

     will

     likely

     grow

    .

     This

     could

     lead

     to

     improved

     patient

     outcomes

     and

     more

     efficient

     use

     of

     healthcare

     resources

    .
    


    2

    .

     Autonomous

     vehicles

    :

     With

     the

     increasing

     use

     of

     AI

     in

     transportation

    ,

     autonomous

     vehicles

     will

     likely

     become

     more

     widespread

    .

     This

     could

     lead

     to

     safer

     and

     more

     efficient

     transportation

     systems

    ,

     and

     potentially

     reduce

     the

     need

     for

     human

     drivers

    .
    


    3

    .

     Automation

     in

     manufacturing

    :

    



```python
llm.shutdown()
```
