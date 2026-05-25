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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.33s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.46it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.46it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.98it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.04it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.20it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.19it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.19it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.19it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=69.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=69.71 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=7168 avail_mem=69.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=6656 avail_mem=69.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=69.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.99it/s]Capturing num tokens (num_tokens=6144 avail_mem=69.70 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=69.69 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=69.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=69.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=69.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=69.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=69.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=69.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=69.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=69.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=69.66 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=69.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=69.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=69.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.65 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=69.65 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=69.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=69.62 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]

    Capturing num tokens (num_tokens=960 avail_mem=69.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s] Capturing num tokens (num_tokens=896 avail_mem=69.64 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=832 avail_mem=69.63 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.80it/s]Capturing num tokens (num_tokens=832 avail_mem=69.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=768 avail_mem=69.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=704 avail_mem=69.63 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=640 avail_mem=69.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=576 avail_mem=69.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=512 avail_mem=69.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.40it/s]Capturing num tokens (num_tokens=512 avail_mem=69.61 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=480 avail_mem=69.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=448 avail_mem=69.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]

    Capturing num tokens (num_tokens=416 avail_mem=69.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=384 avail_mem=69.62 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=352 avail_mem=69.61 GB):  50%|█████     | 29/58 [00:00<00:00, 42.78it/s]Capturing num tokens (num_tokens=352 avail_mem=69.61 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=320 avail_mem=69.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=288 avail_mem=69.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=256 avail_mem=69.60 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=240 avail_mem=69.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=224 avail_mem=69.59 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.34it/s]

    Capturing num tokens (num_tokens=224 avail_mem=69.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=208 avail_mem=69.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=192 avail_mem=69.59 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=176 avail_mem=69.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=160 avail_mem=69.58 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=160 avail_mem=69.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=144 avail_mem=69.58 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=128 avail_mem=69.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=112 avail_mem=69.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.77it/s]Capturing num tokens (num_tokens=96 avail_mem=69.57 GB):  74%|███████▍  | 43/58 [00:01<00:00, 36.77it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=69.57 GB):  81%|████████  | 47/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=80 avail_mem=69.57 GB):  81%|████████  | 47/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=64 avail_mem=69.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=48 avail_mem=69.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=32 avail_mem=69.56 GB):  81%|████████  | 47/58 [00:01<00:00, 34.56it/s]Capturing num tokens (num_tokens=32 avail_mem=69.56 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.73it/s]Capturing num tokens (num_tokens=28 avail_mem=69.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.73it/s]Capturing num tokens (num_tokens=24 avail_mem=69.55 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.73it/s]Capturing num tokens (num_tokens=20 avail_mem=69.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.73it/s]Capturing num tokens (num_tokens=16 avail_mem=69.54 GB):  88%|████████▊ | 51/58 [00:01<00:00, 33.73it/s]

    Capturing num tokens (num_tokens=16 avail_mem=69.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=12 avail_mem=69.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=8 avail_mem=69.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.77it/s] Capturing num tokens (num_tokens=4 avail_mem=69.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.77it/s]Capturing num tokens (num_tokens=4 avail_mem=69.53 GB): 100%|██████████| 58/58 [00:01<00:00, 35.29it/s]


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
    Generated text:  Henry. I'm very shy and always get nervous when I speak in front of others. I feel like I'm always making mistakes and don't know what to say next. I'm really scared of getting scared. 1. How can I become more confident in myself and speak in front of others? 2. Can you give me some tips to help me? 3. What are some other things I can do to stop being afraid of speaking in front of others?
    1. To become more confident in yourself and speak in front of others, you can focus on building your confidence by practicing public speaking skills. You can read and
    ===============================
    Prompt: The president of the United States is
    Generated text:  considered a symbol of democracy and freedom. President Clinton has been described by the New York Times as the best president in the 20th century. When President Clinton died on January 20, 2009, the nation mourned as he left office. In a time when the country was in the throes of a recession, the death of the leader was a cause for celebration. By his death, Clinton was one of the first U. S. presidents to be elected to a second term, and his victory at the 1992 presidential election was a landmark in American democracy. He had been a Republican
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. New York D. Tokyo
    Answer:
    
    A
    
    For the solution of the quadratic equation x^2 - 4x + 4 = 0, which of the following is the correct result for the discriminant Δ?
    A. Δ = -8
    B. Δ = 0
    C. Δ = 8
    D. Δ = 16
    Answer:
    
    B
    
    What is the primary purpose of conducting a drug trial?
    A. To determine the safety and efficacy of drugs
    B. To develop new drugs
    C. To evaluate the side effects of drugs
    D
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright. It is the future of technology and we are already witnessing the impact of AI in many aspects of our daily lives. In the past few years, AI has been used in various fields including healthcare, finance, and transportation. With the continued advancements in AI, we can expect even more exciting developments in the future. AI is still a rapidly evolving field, and there are many exciting areas to explore in the future. As we look ahead, it's important to keep an open mind and be open to new ideas and advancements.
    
    One area where AI has already made a significant impact is in healthcare. AI is being used to develop personalized


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city renowned for its rich history, beautiful architecture, and vibrant culture. It is located in the south of France and is the largest city in the country, with a population of over 10 million people. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe, as well as its diverse cuisine and fashion scene. The city is also home to many world-renowned museums, theaters, and art galleries. Paris is a cultural and economic hub of France and a major tourist destination,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, privacy, and transparency. AI developers will need to be more mindful of how their technology is used and how it impacts society.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making in the future. This will involve the use of AI to assist with
    


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
    Generated text:  [Name], and I’m a versatile and creative individual with a love for art, music, and literature. I’ve always been passionate about exploring the unknown, whether that means diving into the depths of the ocean or exploring the depths of the human psyche. I’ve also been known for my exceptional problem-solving skills, and I find that my ability to think outside the box is one of my greatest strengths.
    
    I’m always looking for new challenges and opportunities to grow and learn, and I’m always happy to help someone who is seeking to learn and grow. I’m constantly looking for new opportunities to create art, music, and literature, and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. 
    
    Here's a more concise version: 
    Paris, France's largest and most important city, serves as the political, economic, and cultural center of the country, renowned for its grandiose architecture, vibrant culture, and diverse population. 
    
    This concise statement captures the essence of Paris as the capital while omitting some details that might be considered less important for a brief description. The statement is directly relevant to the given prompt and provides a clear and concise factual statement about the capital city. 
    
    Additionally, the statement includes a slightly broader context by mentioning it as the "City of Light"
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by increasing integration with other areas of technology, such as machine learning, natural language processing, and computer vision. This integration will likely result in more sophisticated algorithms and models, as well as the ability to learn and adapt to new situations. Additionally, there will be an increased focus on ethical considerations and privacy concerns, as AI systems become more integrated into daily life. Finally, there is a potential for AI to have a positive impact on society, such as through the development of more efficient and sustainable technologies, or through the reduction of human error and exposure to dangerous situations. However, there is also the potential for AI to be


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

    insert

     name

     of

     the

     character

    ].

     I

    'm

     a

     [

    insert

     age

    ]

     year

     old

     [

    insert

     occupation

    ]

     who

     enjoys

     [

    insert

     a

     specific

     hobby

    ,

     interest

    ,

     or

     activity

    ].

     Outside

     of

     work

    ,

     I

     enjoy

     [

    insert

     something

     enjoyable

    ,

     like

     spending

     time

     with

     friends

    ,

     traveling

    ,

     or

     reading

    ].

     I

     have

     a

     passion

     for

     [

    insert

     something

     interesting

    ,

     like

     music

    ,

     literature

    ,

     or

     science

    ].

     How

     can

     you

     say

     you

    're

     "

    neutral

    "?

     I

    'm

     not

     trying

     to

     be

     someone

     else

    .

     I

     just

     seem

     to

     be

     myself

    .

     No

     one

     can

     tell

     what

     I

    'm

     thinking

     or

     feeling

    .

     Just

     an

     average

     person

    ,

     just

     someone

     who

     has

     been

     working

     in

     this

     field

     for

     many

     years

    .

     If

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

    ,

     officially

     known

     as

     the

     City

     of

     Light

    ,

     is

     the

     capital

     city

     of

     France

    .

     It

     is

     situated

     on

     the

     Se

    ine

     River

    ,

     which

     forms

     the

     longest

     river

     in

     Europe

    .

     The

     city

     is

     known

     for

     its

     cultural

     richness

    ,

     including

     its

     iconic

     E

    iff

    el

     Tower

     and

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     population

    ,

     which

     includes

     a

     mix

     of

     French

    ,

     French

     American

    ,

     and

     foreign

     citizens

    .

     The

     city

     is

     renowned

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     art

     scene

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     Paris

     is

     considered

     one

     of

     the

     most

     beautiful

     cities

     in

     the

     world

     and

     is

     home

     to

     many

     important

     landmarks

     and

     historical

     sites

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     significant

     advancements

     in

     several

     key

     areas

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

     Improved

     natural

     language

     processing

    :

     With

     the

     ongoing

     growth

     of

     AI

    ,

     natural

     language

     processing

     (

    N

    LP

    )

     will

     become

     more

     advanced

    .

     This

     will

     enable

     machines

     to

     understand

     and

     interpret

     human

     language

     in

     ways

     that

     are

     previously

     un

    att

    ain

    able

    ,

     leading

     to

     new

     applications

     such

     as

     virtual

     assistants

     and

     chat

    bots

    .
    


    2

    .

     Personal

    ized

     AI

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     personalized

     AI

     solutions

    .

     This

     could

     involve

     using

     AI

     to

     better

     understand

     individual

     users

    '

     preferences

     and

     behaviors

    ,

     leading

     to

     more

     efficient

     and

     effective

     products

     and

     services

    .
    


    3

    .

     Greater

    



```python
llm.shutdown()
```
