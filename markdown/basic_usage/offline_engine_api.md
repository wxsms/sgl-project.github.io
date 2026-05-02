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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.82it/s]


    2026-05-02 14:10:27,356 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 14:10:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:12,  4.44s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.46it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.38it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.38it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.84it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.89it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.91it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.91it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.91it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.91it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.91it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.91it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.14it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]

    Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.66it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=480 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.86it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]

    Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=256 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.62it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.17it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 49.17it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.32it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=28 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  86%|████████▌ | 50/58 [00:01<00:00, 49.08it/s]

    Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.63it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 49.63it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 44.31it/s]


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
    Generated text:  Lisa. My name is Lisa Williams. I'm a nice teacher at the school. I have a new classmate. His name is Andy. He is from America. I like him very much. We are the same age. I like talking to him and listening to him. Andy is very good at swimming. He can swim very well. He likes swimming. He can swim well. He can swim very fast and jump very high. We often go swimming. We all like to watch the swimming. We go swimming at the swimming club. When we are free, we often go to the swimming club to swim. We have a big
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 38 years old. How old will the president be in 7 years if he is currently 38 years old? The president is currently 38 years old. If he is currently 38 years old, in 7 years, he will be 38 + 7 = 45 years old. Answer: 45.
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Berlin
    D. Moscow
    答案：A
    分析题干，巴黎是法国的首都，选项A正确。答案选A。北京是中华人民共和国的首都，选项B错误。伦敦是英国的首都，选项C错误。莫斯科是俄罗斯的首都，选项D错误。本题答案选A。
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain but it has the potential to transform the way we work, learn, and communicate. But it also brings challenges and questions about how we will govern and protect this technology. The complexity of the issue also makes it hard to come to a consensus on a solution. In this article, we discuss how AI can be managed to make sure that it is used responsibly.
    1. AI and Human Rights
    The issue of AI can be complicated by the fact that it is still in the early stages of development and regulation is not yet in place. There is some debate about the potential risks and benefits of AI, but there is also a need to


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn and take on new challenges. What do you do for a living? As a [job title], I'm always looking for ways to improve my skills and knowledge. I'm always eager to learn and take on new challenges. What are your hobbies? As a [job title], I'm always looking for ways to improve my skills
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is also home to the world's largest city, with over 10 million people living in the city. The city is known for its diverse population, including French, African, and immigrant communities. Paris is a popular tourist destination, with millions of visitors annually. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks, from simple tasks like language translation to complex tasks like autonomous driving and medical diagnosis. Additionally, AI will continue to be integrated into various industries, from manufacturing and healthcare to finance and transportation, leading to increased efficiency and productivity. However, there are also potential risks and challenges associated with AI, such as the potential for job displacement and the need for ethical and responsible development and deployment of AI systems. Overall, the future
    


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
    Generated text:  [Your Name]. I'm a [Your occupation or role]. I love [Your passion or hobby]. I'm a [Your location] resident, [Your age], and I'm [Your favorite color, or a characteristic you're particularly proud of]. I like to [Your hobby or activity], and I enjoy [Your interests or hobbies]. In my free time, I enjoy [Your interests or hobbies]. And my favorite movie is [Your favorite movie or series]. I'm also a [Your occupation or role], and I've always loved [Your favorite book or piece of art]. And I enjoy [Your favorite book or piece of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known as the "City of Light" due to its famous skyline, rich history, and vibrant culture. 
    
    (Note: The statement above is a factual fact about Paris and not a question or response to an inquiry.) 
    
    Facts about Paris: 
    - It is the capital city of France.
    - Its nickname is "City of Light."
    - It is known for its stunning architecture, including the Eiffel Tower.
    - Paris is known for its rich history and cultural significance.
    - It is also known for its important role in world events, including the World Cup.
    
    Paris is a historical and cultural capital of France,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a topic of ongoing research and discussion, with many different paths and potential futures. Some possible trends include:
    
    1. Increased automation and machine learning: With the development of AI technologies, we can expect to see more automation and machine learning applications in various industries. This could lead to the creation of new jobs in areas like data analysis, software engineering, and hardware design.
    
    2. Enhanced human-computer interaction: AI is becoming more integrated into our daily lives, from our smart home devices to our virtual assistants. As AI continues to improve, we may see a shift towards more human-computer interaction, with AI becoming more accessible and intuitive.
    
    3


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

     am

     a

     [

    Role

    ]

     in

     the

     [

    Company

    /

    Industry

    ].

     I

     started

     [

    Company

    /

    Industry

    ]

     [

    Year

    ]

     as

     a

     [

    Position

    ].

     I

    've

     always

     been

     passionate

     about

     [

    interest

     or

     hobby

    ].

     My

     journey

     as

     a

     [

    Company

    /

    Industry

    ]

     professional

     has

     been

     filled

     with

     challenges

     and

     successes

    .

     I

     enjoy

     [

    What

     I

     do

     well

    ].

     How

     would

     you

     describe

     yourself

    ?


    [

    Name

    ]:

     I

     am

     [

    Type

    ]

     and

     I

     am

     passionate

     about

     [

    my

     interest

     or

     hobby

    ].

     I

     believe

     in

     [

    Why

     I

     love

     my

     job

     or

     industry

    ].

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

     strive

     for

    ].

     I

    'm

     [

    What

     I

     hope

     to

     achieve

     in

    
    
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

    .

     It

     is

     a

     historical

     and

     cultural

     city

     located

     in

     the

     north

    western

     part

     of

     the

     country

     and

     is

     the

     largest

     metropolitan

     area

     in

     Europe

    .

     The

     city

     is

     known

     for

     its

     rich

     history

    ,

     world

    -ren

    owned

     museums

     and

     art

     galleries

    ,

     and

     its

     vibrant

     nightlife

    .

     Paris

     is

     also

     famous

     for

     its

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     a

     popular

     tourist

     destination

     and

     an

     important

     economic

     and

     political

     center

     in

     France

    .

     
    


    Some

     key

     facts

     about

     Paris

    :
    


    -

     It

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    ,

     the

     world

    's

     tallest

     iron

     lattice

     structure

    .


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     there

     is

     no

     single

    ,

     clear

     trend

     or

     path

     that

     will

     lead

     to

     a

     perfect

     future

    .

     However

    ,

     there

     are

     several

     key

     trends

     that

     are

     likely

     to

     shape

     the

     future

     of

     AI

    ,

     including

    :
    


    1

    .

     Increased

     development

     of

     neural

     networks

     and

     machine

     learning

     algorithms

    :

     Neural

     networks

     and

     machine

     learning

     algorithms

     are

     expected

     to

     continue

     to

     improve

     and

     become

     more

     powerful

     over

     time

    .

     This

     will

     lead

     to

     better

    理解和

    output

     for

     a

     wide

     range

     of

     tasks

    ,

     including

     natural

     language

     processing

     and

     computer

     vision

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

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

     such

     as

     blockchain

    ,

     quantum

     computing

    ,

     and

     edge

     computing

    .

     This

     will

     make

     it

     easier

     for

     AI

    



```python
llm.shutdown()
```
