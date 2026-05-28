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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.95it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.98s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.98s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:47,  3.98s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.83it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.83it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.21it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.82it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.23it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.91it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.01it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s] Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.12it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.61it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=384 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 44.07it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.71it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.95it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.95it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.95it/s]Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.95it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.95it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.73it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.68it/s]Capturing num tokens (num_tokens=20 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=16 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.96it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.96it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 42.25it/s]


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
    Generated text:  Reza. I have recently started studying at the University of London, where I'm currently studying in Computer Science. I'm excited to see how my studies will develop my skills. How can I improve my English proficiency? Improving your English proficiency can be achieved through a variety of methods, including practicing with native speakers, reading extensively, listening to English-language media, watching English-language movies and TV shows, and using online resources. Additionally, you can take English language courses or enroll in English language tutoring services if you're having trouble improving your language skills. Overall, the key is consistent effort and dedication. Good luck! Best regards, Re
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to go to war with another country. The president and his advisers have decided to look at a complex matrix of information to help make their decision. They are using the game theory framework to model the situation.
    
    The matrix is as follows:
    
    \[
    \begin{array}{c|cc}
     & \text{Canada} & \text{United States} \\
    \hline
    \text{Canada} & 20 & 0 \\
    \text{United States} & 5 & 20 \\
    \end{array}
    \]
    
    In this matrix, the first number in each cell represents the amount of
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Rome C. Brussels D. London
    A. Paris
    
    The capital of France is Paris. 
    
    B. Rome, C. Brussels, and D. London are cities located in the same countries.
    
    Paris is the largest city in France and the capital of France. It is also the seat of the French government and the seat of the French parliament. The city is famous for its rich history, architecture, and culture, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. 
    
    While Rome, Brussels, and London are all cities in Europe, London is the largest city in
    ===============================
    Prompt: The future of AI is
    Generated text:  not to replace humans, but to augment them.
    
    How can AI be used to enhance human intelligence?
    
    AI can be used to augment human intelligence in various ways, such as:
    
    1. Improving language understanding: AI can process and analyze large volumes of text and speech data, allowing computers to understand and interpret human language more accurately and comprehensively.
    
    2. Enhancing decision-making: AI algorithms can analyze and analyze vast amounts of data in real-time, providing insights and recommendations that can help humans make more informed and data-driven decisions.
    
    3. Improving healthcare: AI can assist doctors in diagnosing diseases, predicting patient outcomes, and even helping


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number of Wheels] wheels. I'm [Favorite Color] and I love [Favorite Activity]. I'm [Favorite Book] and I enjoy [Favorite Food]. I'm [Favorite Movie] and I love [Favorite Music]. I'm [Favorite Sport] and I play [Favorite Sport]. I'm [Favorite Place] and I love [Favorite Thing]. I'm [Favorite Animal] and I have [Number of Paws] paws. I'm [Favorite Book] and I enjoy [Favorite Food].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also the birthplace of many famous French artists and writers. The city is known for its cuisine, including its famous croissants, and its fashion industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, with its diverse
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and personalized medicine to virtual assistants and chatbots. As AI becomes more integrated into our daily lives, it is likely to have a significant impact on the way we work, communicate, and interact with each other. However, it is also important to consider the potential risks and ethical considerations associated with AI, and to work to develop and implement policies and regulations that ensure its safe and responsible use. Ultimately, the future of AI is
    


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
    Generated text:  [Your Name], and I am a [Your Profession] who is [Your Career Objective]. I am passionate about [Your Passion], and I strive to make a positive impact in the world through my work and personal life. I am a [Your Interests] who enjoy [Your Activities]. I believe that being an [Your Job Title] is essential to [Your Profession], and I am eager to [Your Goal or Cause]. I am a [Your Job Title] who is always [Your Career Objective] and [Your Interests]. I am a [Your Career Objective] who is always [Your Interests] and [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with a population of over 2.7 million.
    
    Paris is the capital city of France, located on the Seine River in the Loire Valley. It is the largest and most populous city in France by population, with an estimated population of 2.7 million as of the 2019 census. The city is known for its rich history, diverse cultural scene, and iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. Paris is a major transportation hub for the country and hosts numerous international events and conferences. It is often referred to as the "
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  undoubtedly going to see significant and diverse changes as technology advances and we move toward a more interconnected world. Here are some possible future trends in AI:
    
    1. Augmented Intelligence: AI is not just about creating intelligent machines, but it's also about making them more human-like. Augmented intelligence refers to the ability of AI to interact with humans in a human-like way. This could be achieved through the use of virtual and augmented reality technologies, where AI is used to create or enhance the human experience.
    
    2. Ethical AI: As AI is becoming more integrated into various aspects of our lives, there is a growing concern about its ethical implications


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

     John

    .

     I

    'm

     a

     

    3

    5

    -year

    -old

     software

     engineer

     with

     a

     passion

     for

     technology

     and

     innovation

    .

     My

     love

     for

     coding

     has

     taken

     me

     to

     some

     of

     the

     most

     prestigious

     tech

     companies

     in

     the

     world

    ,

     where

     I

    've

     had

     the

     privilege

     of

     contributing

     to

     the

     latest

     projects

     and

     developing

     groundbreaking

     solutions

    .

     I

    'm

     always

     eager

     to

     learn

     new

     things

     and

     seek

     opportunities

     to

     grow

     and

     improve

     my

     skills

    .

     Outside

     of

     work

    ,

     I

     enjoy

     spending

     time

     with

     my

     family

     and

     exploring

     my

     love

     for

     hiking

     and

     photography

    .

     Thank

     you

     for

     considering

     me

     as

     a

     potential

     partner

    !

     [

    insert

     

    3

    0

    -

    6

    0

     seconds

     of

     conversation

     with

     John

    ]

     John

    :

     Hi

    ,

     I

    'm

     impressed

     by

     your

     passion

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     statement

     can

     be

     summarized

     as

    :

     Paris

     is

     the

     largest

     city

     in

     France

     and

     one

     of

     the

     most

     influential

     cities

     in

     Europe

    .

     This

     fact

     encaps

    ulates

     the

     capital

    's

     status

     as

     a

     major

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

    .

     
    


    Here

    's

     a

     slightly

     refined

     version

     for

     clarity

    :

     
    


    -

     The

     French

     capital

     is

     Paris

    ,

     making

     it

     the

     most

     populous

     city

     in

     France

    .


    -

     Paris

    ,

     with

     its

     rich

     history

    ,

     cultural

     heritage

    ,

     and

     influence

     over

     Europe

    ,

     is

     considered

     the

     largest

     and

     most

     influential

     city

     in

     France

    .

     
    


    This

     statement

     provides

     a

     succinct

     overview

     of

     Paris

    '

     importance

     within

     the

     broader

     context

     of

     French

     and

     European

     affairs

    .

     Let

     me

     know

     if

     you

    'd

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     is

     expected

     to

     become

     more

     integrated

     with

     human

     intelligence

    ,

     enabling

     more

     sophisticated

     and

     complex

     decision

    -making

    .

     This

     will

     require

     significant

     advances

     in

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     other

     areas

     of

     AI

     research

    .
    


    2

    .

     Natural

     language

     processing

     (

    N

    LP

    ):

     As

     AI

     systems

     become

     more

     complex

    ,

     N

    LP

     is

     likely

     to

     become

     a

     key

     driver

     of

     AI

     innovation

    .

     N

    LP

     will

     enable

     AI

     systems

     to

     understand

     and

     interpret

     human

     language

     in

     a

     more

     nuanced

     and

     sophisticated

     way

    .
    


    3

    .

     Increased

     use

     of

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

    



```python
llm.shutdown()
```
