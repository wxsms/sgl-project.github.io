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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:45,  3.96s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:45,  3.96s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.84it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.77it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.16it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.68it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.43it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=53.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=53.65 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=53.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=5120 avail_mem=53.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4608 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.62 GB):   9%|▊         | 5/58 [00:00<00:02, 21.94it/s]Capturing num tokens (num_tokens=4096 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3840 avail_mem=53.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3584 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.59 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s]Capturing num tokens (num_tokens=960 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s]Capturing num tokens (num_tokens=832 avail_mem=53.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.09it/s]Capturing num tokens (num_tokens=832 avail_mem=53.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=768 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=704 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=640 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=576 avail_mem=53.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=53.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.37it/s]Capturing num tokens (num_tokens=512 avail_mem=53.55 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]Capturing num tokens (num_tokens=480 avail_mem=53.57 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]Capturing num tokens (num_tokens=448 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]Capturing num tokens (num_tokens=416 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]

    Capturing num tokens (num_tokens=384 avail_mem=53.56 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]Capturing num tokens (num_tokens=352 avail_mem=53.55 GB):  50%|█████     | 29/58 [00:00<00:00, 42.51it/s]Capturing num tokens (num_tokens=352 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=320 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=288 avail_mem=53.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=256 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=240 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=224 avail_mem=53.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.13it/s]Capturing num tokens (num_tokens=224 avail_mem=53.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=208 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=192 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=176 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=53.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=144 avail_mem=53.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=144 avail_mem=53.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=128 avail_mem=53.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=112 avail_mem=53.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=96 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s] Capturing num tokens (num_tokens=80 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=64 avail_mem=53.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.72it/s]Capturing num tokens (num_tokens=64 avail_mem=53.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=48 avail_mem=53.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=32 avail_mem=53.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=28 avail_mem=53.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]

    Capturing num tokens (num_tokens=24 avail_mem=53.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=20 avail_mem=53.49 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.69it/s]Capturing num tokens (num_tokens=20 avail_mem=53.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=16 avail_mem=53.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=12 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=8 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.86it/s] Capturing num tokens (num_tokens=4 avail_mem=53.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.86it/s]Capturing num tokens (num_tokens=4 avail_mem=53.48 GB): 100%|██████████| 58/58 [00:01<00:00, 40.87it/s]


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
    Generated text:  Christine. I'm a high school senior who has been experiencing a lot of stress lately. I feel like I can't find my way home in the morning and my emotions are quite high. I'm not sure if I'm going to have the strength to deal with this stress. I'm a bit of a perfectionist and always looking for perfection, which is a lot for me to handle. 
    
    I've been struggling with anxiety, I'm having difficulty sleeping, and my mood is pretty dark. I'm trying my best to focus on the things that are truly important to me and the things that are important to my family. I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected official in the executive branch of the government of the United States. He or she holds the office of the presidency, which is a position in the government of the United States. The president is in charge of the United States government. The president is the leader of the federal government in the United States. The president is also the Commander-in-Chief of the United States Armed Forces.
    Does this next sentence follow, given the preceding text?
    The President is not the person who serves as the leader of the federal government.
    
    Pick your answer from:
    (a). yes.
    (b). no.
    (a). Yes
    You are a helpful assistant with
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in Europe, the second largest in the world and the most populous city in the European Union. With a population of around 2.1 million as of 2017, Paris is the most populous city in France and the second-most populous city in the European Union.
    The city has been inhabited since the Roman Empire, and has been the capital of France since the end of the Hundred Years' War, in 1804. It has been the capital of the French Republic since 1792. In 2017, the city had a population of 1
    ===============================
    Prompt: The future of AI is
    Generated text:  HERE, not there. Let's discuss the emerging trends in AI that will shape the future of the industry and the world. Join us for an informative discussion on the future of AI that covers topics such as machine learning, deep learning, natural language processing, computer vision, robotics, and more. Who is this event aimed at? We aim to provide insights and discussions on the future of AI that will shape the industry and the world. Who is this event open to? This event is open to anyone interested in the future of AI and its impact on various industries. What will be the main focus of the event? The main focus of the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling city with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with many famous designers and boutiques located in the city. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role in shaping French culture and identity. The city is also home to many
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI is already being used to automate a wide range of tasks, from manufacturing to customer service. As AI technology continues to improve, we can expect automation to become more widespread and efficient.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be an increased need for privacy and security measures to protect user data. This will likely lead to more stringent regulations and standards for AI development and use.
    
    3. AI ethics and responsibility: As AI systems become more complex and autonomous, there will be a growing need for ethical guidelines and responsibility to be placed on
    


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
    Generated text:  [Name], and I am a [type of person] person. I enjoy [occupation], and I love to [describe your passion or hobby]. I am currently [age] years old, and I currently reside in [your current city or state]. I am very [positive] and I believe in [thing or principle]. I have a [favorite hobby or activity], and I believe that [explain why you enjoy it]. I value [other qualities or traits] and I hope to [describe your future aspirations or goals]. I am [independent] and I am always [positive] about my life. I am [able to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is located on the left bank of the Seine river and is known for its historical significance, iconic architecture, and vibrant cultural scene. Paris is a major global city, with millions of people living in the city and attracting visitors from all over the world. It is home to many world-renowned landmarks and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is known for its cuisine, fashion, and music, as well as its cultural events and festivals. Paris has a rich history dating back thousands of years and continues to be a center of culture, art, and innovation
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating and diverse, with a wide range of trends that are shaping its trajectory. Some potential future trends in AI include:
    
    1. Increased use of AI in healthcare: With advances in AI, we can expect to see more personalized and accurate medical treatments in the future. AI can help doctors analyze patient data, identify potential health risks, and develop personalized treatment plans.
    
    2. Autonomous vehicles: With the increasing reliance on AI in vehicles, we can expect to see more autonomous vehicles on the roads in the future. Autonomous vehicles can help reduce traffic accidents, improve safety, and enhance transportation efficiency.
    
    3. AI in manufacturing: AI is already being used


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

    ].

     I

    'm

     a

     professional

     freelance

     writer

     and

     editor

     with

     over

     

    1

    0

     years

     of

     experience

     in

     the

     industry

    .

     I

     specialize

     in

     writing

     and

     editing

     technical

     and

     business

     documents

    ,

     including

     reports

    ,

     journals

    ,

     and

     presentations

    .

     I

     have

     a

     passion

     for

     creativity

     and

     my

     work

     is

     consistently

     praised

     for

     my

     ability

     to

     convey

     complex

     information

     in

     a

     clear

     and

     concise

     manner

    .

     I

    'm

     always

     eager

     to

     learn

     and

     adapt

     to

     new

     approaches

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     in

     the

     industry

    .

     I

    'm

     confident

     that

     my

     skills

     and

     experience

     make

     me

     a

     valuable

     asset

     to

     any

     organization

     looking

     for

     a

     skilled

     and

     experienced

     writer

     and

     editor

    .

     Thank

     you

     for

     considering

     me

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     city

     with

     the

     most

     population

     in

     France

     and

     the

     world

    's

     largest

     city

    .

     It

     is

     located

     on

     the

     Se

    ine

     river

     and

     is

     the

     largest

     city

     in

     Europe

     by

     land

     area

    ,

     with

     an

     estimated

     

    1

    .

    3

     million

     inhabitants

     (

    as

     of

     

    2

    0

    1

    9

    ).

     It

     is

     also

     home

     to

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

     is

     known

     for

     its

     rich

     history

    ,

     architecture

    ,

     culture

    ,

     and

     cuisine

    .

     The

     city

     is

     also

     home

     to

     many

     famous

     landmarks

     such

     as

     the

     Lou

    vre

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Ch

    amps

    -

    É

    lys

    ées

    .

     It

     is

     an

     important

     cultural

     and

     economic

     center

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     diverse

    ,

     and

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

     and

     applications

     of

     the

     field

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

     Increased

     use

     of

     AI

     in

     healthcare

    :

     With

     the

     increasing

     availability

     of

     large

     amounts

     of

     data

     and

     the

     development

     of

     AI

     algorithms

     that

     can

     analyze

     this

     data

    ,

     there

     is

     potential

     for

     AI

     to

     play

     a

     role

     in

     healthcare

    .

     AI

     could

     be

     used

     to

     diagnose

     diseases

    ,

     predict

     patient

     outcomes

    ,

     and

     even

     assist

     in

     drug

     discovery

    .
    


    2

    .

     AI

     in

     manufacturing

    :

     AI

     can

     be

     used

     to

     optimize

     production

     processes

    ,

     reduce

     costs

    ,

     and

     improve

     quality

    .

     For

     example

    ,

     AI

     algorithms

     could

     be

     used

     to

     predict

     equipment

     failures

     or

     to

     optimize

    



```python
llm.shutdown()
```
