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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.20it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:13,  3.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:04,  8.15it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]

    Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:04<00:04,  8.15it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]

    Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:04<00:01, 20.23it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.06it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]

    Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 33.71it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   3%|▎         | 2/58 [00:00<00:03, 16.56it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:03, 16.56it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:03, 16.56it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:03, 16.56it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   9%|▊         | 5/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):   9%|▊         | 5/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.66 GB):   9%|▊         | 5/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):   9%|▊         | 5/58 [00:00<00:02, 19.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.08it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=55.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.64 GB):  21%|██        | 12/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.63 GB):  21%|██        | 12/58 [00:00<00:01, 26.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.63 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.65it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.62 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s] Capturing num tokens (num_tokens=896 avail_mem=55.61 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=832 avail_mem=55.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.13it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=704 avail_mem=55.60 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=576 avail_mem=55.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=512 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.40it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=416 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:00<00:00, 37.25it/s]Capturing num tokens (num_tokens=384 avail_mem=55.59 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.25it/s]Capturing num tokens (num_tokens=320 avail_mem=55.58 GB):  52%|█████▏    | 30/58 [00:01<00:00, 37.25it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.58 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=288 avail_mem=55.58 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=224 avail_mem=55.57 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  60%|██████    | 35/58 [00:01<00:00, 38.38it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=176 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=160 avail_mem=55.56 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.13it/s]Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s] Capturing num tokens (num_tokens=80 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=64 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  78%|███████▊  | 45/58 [00:01<00:00, 39.75it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=32 avail_mem=55.53 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  86%|████████▌ | 50/58 [00:01<00:00, 40.41it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.05it/s] Capturing num tokens (num_tokens=4 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:01<00:00, 41.05it/s]Capturing num tokens (num_tokens=4 avail_mem=55.51 GB): 100%|██████████| 58/58 [00:01<00:00, 35.55it/s]


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
    Generated text:  Alex, and I am a fan of the Marvel Universe. Is there anything you can tell me about the Avengers? Sure! The Avengers is a superhero team from Marvel Comics. They are a diverse group of superheroes who work together to protect Earth from a cosmic threat known as the X-Index. The team includes Spider-Man, Iron Man, Captain America, Hulk, Black Widow, and Thor. They are known for their speed, strength, and teamwork, and they have battled various villains throughout the Marvel Universe. 
    
    Is there anything else you would like to know about the Avengers? Let me know! 
    
    What a fun topic for a fan
    ===============================
    Prompt: The president of the United States is
    Generated text:  a prime minister of the United States of America. If the president is the prime minister of the United States, does it mean that he is also a prime minister of the United States of America? Yes, if the president is the prime minister of the United States of America, it does not necessarily mean that he is also a prime minister of the United States. The president is appointed by the President of the United States, who is the head of the executive branch, and is responsible for the country's policy and overall direction. While the president is the head of the executive branch, they are not necessarily the head of government or the head of
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Ottawa
    D. Amsterdam
    Answer:
    A
    
    The purpose of the police is to protect the public and safeguard the interests of the state.
    A. Correct
    B. Incorrect
    Answer:
    A
    
    [Multiple Choice Question] The most suitable time for childbirth is ____.
    A. 3-4 days
    B. 5-6 days
    C. 7-8 days
    D. 10-12 days
    E. 14 days
    Answer:
    7-8 days
    
    Which of the following statements is true? A. The theory of
    ===============================
    Prompt: The future of AI is
    Generated text:  developing rapidly and the impact on human society is growing. As artificial intelligence becomes more advanced, the need for better ways to regulate its use has grown. The field of AI regulation is evolving, and it includes a range of strategies to ensure that the benefits of AI are distributed fairly and that the risks are mitigated. One approach to AI regulation is to ensure that the AI is developed with human oversight to prevent unintended consequences.
    
    One of the main challenges in AI regulation is ensuring that the AI is developed with human oversight. This can be difficult due to the nature of AI, which involves complex decision-making and decision-making that is often beyond the ability


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


    Generated text:  [Name] and I am a [Age] year old [Gender] [Occupation]. I have always been fascinated by [Subject] and have always wanted to learn more about it. I have always been a [Skill] person and have always been interested in [Interest]. I am always looking for new experiences and have always been open to trying new things. I am always eager to learn and grow, and I am always looking for ways to improve myself. I am a [Personality] person and I am always looking for ways to make the world a better place. I am always looking for new challenges and opportunities to grow and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament and the French Parliament building. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. The city is known for its cuisine, fashion, and music, and is a popular tourist destination. Paris is also home to many international organizations and institutions, including the European Parliament and the United Nations. The city is known for its vibrant nightlife and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in various industries, including manufacturing, transportation, and healthcare. Automation will likely lead to the development of more efficient and cost-effective systems, which will enable businesses to increase their productivity and competitiveness.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its impact on society. There will be a need for regulations and guidelines to ensure that AI is used ethically and
    


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
    Generated text:  [Name], and I'm a [Age] year old [Gender] [Occupation] [Description of Character]. I am... [Describe your character's personality traits, strengths, weaknesses, and any other relevant qualities]. How would you like to meet you?
    As an AI language model, I don't have a physical appearance or a personality. However, I can help you create a neutral self-introduction for a fictional character based on your description. Can you please provide me with some information about your character? I can use that information to create a self-introduction that is appropriate for your role. Let me know! [Describe your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light. It is the most populous city in France and the world, with a population of over 2 million residents. Paris is home to many world-renowned landmarks, including Notre-Dame Cathedral, the Eiffel Tower, the Louvre Museum, and the Louvre Pyramid. The city is also renowned for its diverse culture and cuisine, as well as its rich history and cultural heritage. Paris is a major city with a rich cultural history and a strong sense of identity, making it an important cultural and economic center in France and the world. The city is known for its elegance, sophistication,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be unpredictable, but some possible trends that could shape the technology in the years to come include:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is expected to become more integrated with human intelligence, enabling machines to learn from and adapt to human behavior and experience. This could lead to more sophisticated, adaptable AI that can make more accurate predictions and decisions.
    
    2. Expansion of AI into more sectors: AI is already being used in a wide range of industries, from healthcare and finance to transportation and manufacturing. It is expected to continue expanding into new sectors, such as manufacturing and retail, as technology becomes more integrated


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

    ].

     I

     am

     a

     [

    Career

     Objective

    /

    Current

     Role

    ]

     at

     [

    Company

    ].

     I

     recently

     graduated

     from

     [

    Your

     School

    ]

     and

     have

     been

     working

     hard

     to

     achieve

     my

     goals

    .

     I

     am

     confident

     in

     my

     abilities

     and

     look

     forward

     to

     contributing

     to

     [

    Company

    ]

     in

     a

     supportive

     manner

    .

     How

     can

     I

     be

     of

     assistance

     to

     you

    ?

     [

    Your

     Name

    ]

     [

    Career

     Objective

    /

    Current

     Role

    ]

     [

    Company

    ]

     Hello

    ,

     my

     name

     is

     [

    Your

     Name

    ].

     I

     am

     a

     [

    Career

     Objective

    /

    Current

     Role

    ]

     at

     [

    Company

    ].

     I

     recently

     graduated

     from

     [

    Your

     School

    ]

     and

     have

     been

     working

     hard

     to

     achieve

     my

     goals

    .

     I

     am

     confident

     in

     my

     abilities

     and

     look

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     is

     the

     cultural

    ,

     political

    ,

     and

     economic

     heart

     of

     the

     country

    .

     It

     is

     home

     to

     many

     of

     France

    's

     most

     important

     historical

     and

     cultural

     landmarks

     and

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre

     Dame

     Cathedral

    ,

     Lou

    vre

     Museum

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     culinary

     culture

    ,

     fashion

     industry

    ,

     and

     fashion

    -forward

     fashion

     designers

    .

     Paris

     is

     also

     a

     major

     financial

     center

     and

     has

     been

     for

     decades

    ,

     and

     continues

     to

     be

     a

     major

     hub

     for

     many

     aspects

     of

     France

    's

     economic

     and

     political

     life

    .

     Paris

    '

     population

     is

     around

     

    2

    .

     

    5

     million

    ,

     and

     it

    's

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     influenced

     by

     several

     emerging

     trends

    ,

     including

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

     and

     healthcare

     delivery

    :

     With

     the

     rise

     of

     remote

     patient

     monitoring

     and

     tele

    health

    ,

     AI

     can

     help

     healthcare

     providers

     make

     more

     informed

     decisions

     and

     provide

     better

     care

    .

     AI

     can

     also

     be

     used

     to

     predict

     patient

     outcomes

     and

     provide

     personalized

     treatment

     plans

    .
    


    2

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     AI

     can

     be

     used

     to

     improve

     the

     efficiency

     and

     accuracy

     of

     language

     processing

    ,

     such

     as

     in

     chat

    bots

     and

     virtual

     assistants

    .

     This

     can

     lead

     to

     more

     natural

     language

     interactions

     and

     a

     better

     user

     experience

    .
    


    3

    .

     Integration

     with

     more

     advanced

     technologies

    :

     AI

     will

     likely

     become

     more

     integrated

     with

     other

     technologies

    ,

     such

    



```python
llm.shutdown()
```
