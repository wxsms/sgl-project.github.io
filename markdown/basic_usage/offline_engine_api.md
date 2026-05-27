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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.85it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.84it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.51it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.54it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.65it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.00it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 32.23it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 18.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.76it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.76it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.76it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.76it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.76it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  21%|██        | 12/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.72 GB):  21%|██        | 12/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.72 GB):  21%|██        | 12/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.71 GB):  21%|██        | 12/58 [00:00<00:01, 24.19it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.71 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.69 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.21 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=71.12 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.05 GB):  26%|██▌       | 15/58 [00:00<00:01, 24.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.05 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=960 avail_mem=71.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s] Capturing num tokens (num_tokens=896 avail_mem=71.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=832 avail_mem=71.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 29.72it/s]Capturing num tokens (num_tokens=768 avail_mem=71.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]Capturing num tokens (num_tokens=704 avail_mem=71.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]Capturing num tokens (num_tokens=640 avail_mem=71.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]Capturing num tokens (num_tokens=576 avail_mem=71.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 35.08it/s]Capturing num tokens (num_tokens=480 avail_mem=71.03 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=448 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=416 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.94it/s]Capturing num tokens (num_tokens=384 avail_mem=71.02 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=352 avail_mem=71.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 38.94it/s]Capturing num tokens (num_tokens=320 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=288 avail_mem=71.01 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=256 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=240 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.00 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  60%|██████    | 35/58 [00:01<00:00, 41.78it/s]Capturing num tokens (num_tokens=208 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=192 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=176 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=160 avail_mem=70.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=144 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 43.69it/s]Capturing num tokens (num_tokens=128 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=112 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=96 avail_mem=70.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s] Capturing num tokens (num_tokens=80 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s]

    Capturing num tokens (num_tokens=64 avail_mem=70.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.34it/s]Capturing num tokens (num_tokens=48 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=32 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=28 avail_mem=70.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=24 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=20 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 45.09it/s]Capturing num tokens (num_tokens=16 avail_mem=70.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=12 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.56it/s]Capturing num tokens (num_tokens=8 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.56it/s] Capturing num tokens (num_tokens=4 avail_mem=70.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 45.56it/s]

    Capturing num tokens (num_tokens=4 avail_mem=70.94 GB): 100%|██████████| 58/58 [00:01<00:00, 37.36it/s]


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
    Generated text:  Evan. I am an American citizen born in the United States. I am a part of the third generation of African American. My family is from the rural community of Nacogdoches, Texas. We have been living in Texas for more than 150 years and the older generations of our family have been farmers and ranchers. As a child, I enjoyed to play sports. I became a sports enthusiast when I was in college. After graduation, I started my career as a sports agent. I have been involved with sports for almost two decades. I have been a sports analyst and a sports broadcaster since 200
    ===============================
    Prompt: The president of the United States is
    Generated text:  a person; therefore, every president is a person.
    This syllogism is an example of which type of reasoning?
    A. Affirmative reasoning
    B. Dilemma
    C. Modus tollens
    D. Ad Hominem
    E. Absurdity
    To determine the type of reasoning in the given statement, let's break it down step by step.
    
    1. **Identify the given statement:**
       "Every president is a person."
    
    2. **Analyze the structure of the syllogism:**
       The syllogism typically follows the form:
       \[
       \text
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is over 2.5 million people. Paris is the capital of France, and is also the capital of the overseas department of the French Republic. The capital of the French Republic is Paris. The population of the capital is over 2.5 million people. The population of the capital is over 2.5 million. The population of the capital is over 2.5 million. The population of the capital is over 2.5 million. The population of the capital is over 2.5 million. The population of the capital is over 2.5 million. The population of the
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it’s going to change the world. In a world that has become connected, intertwined and interconnected, digital connections are replacing physical ones. Because of this, we have the potential to create a new type of AI which is completely different from what we know of today. It’s the type of AI that can understand and produce art, design, technology, and much more. The more we continue to embrace and use this new type of AI, the more we can expect to see this new world of AI emerge.
    What is AI?
    AI refers to Artificial Intelligence, a science, technology, and engineering that is focused on building computer


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love to Do], and I'm always looking for ways to [What I Want to Improve]. I'm a [What I Like to Do] and I'm always [What I Like to Do]. I'm a [What I Like to Do] and I'm always [What I Like to Do]. I'm a [What I Like to Do] and I'm always [What I Like to Do]. I'm a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country. It is located on the Seine River and is the seat of government, administration, and culture for the French Republic. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, as well as its rich history and diverse culture. The city is also home to many famous museums, including the Louvre and the Musée d'Orsay, and is a popular tourist destination. Paris is a vibrant and dynamic city with a rich cultural heritage and a strong sense of identity. Its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we can expect to see more automation and AI in various industries, from manufacturing and transportation to healthcare and finance. This will lead to increased efficiency, productivity, and cost savings for businesses and individuals alike.
    
    2. Personalization and customization: AI will enable more personalized and customized experiences for users, from personalized recommendations to targeted advertising. This will lead to a more efficient and effective use of resources, as well as a
    


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
    Generated text:  [Name], and I'm a [occupation] from [age]. I enjoy [reason for hobby or interest]. What is your name? My name is [Name]. I'm a [occupation] from [age]. I enjoy [reason for hobby or interest]. [Tell us about yourself, including your hobbies, interests, and any unique qualities that distinguish you from others in your field. For example, if you're a musician, mention your favorite instrument, any singing or dancing, or your unique vocal style.] [Tell us about your long-term goals and aspirations, including any challenges or obstacles you face in achieving them. For example,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the most populous city in France, with an estimated population of 2.1 million as of 2021. The city is the capital of the French Department of Paris, and serves as the political, economic, cultural, and administrative center of France. It is also home to the headquarters of the French government, major cultural institutions, and the main attractions of the country, including the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is known for its rich history, classical architecture, and vibrant cultural scene. The city is also a major transportation
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and will depend on a variety of factors, including technological advancements, societal changes, and policy decisions. However, some possible future trends in AI include:
    
    1. Increased Automation: As AI continues to evolve, we are likely to see even more automation in various industries. For example, AI-powered robots and autonomous vehicles will become more prevalent, reducing the need for human intervention in many aspects of daily life.
    
    2. AI ethics and Privacy: As AI becomes more integrated into our lives, we are likely to face new ethical and privacy issues. As AI systems become more sophisticated, we may need to develop new ways to regulate their use and


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

     first

     name

    ]

     and

     I

    'm

     [

    insert

     first

     name

    ].

     I

    'm

     a

     [

    insert

     profession

     or

     area

     of

     interest

    ]

     with

     a

     [

    insert

     interest

     or

     expertise

    ].

     I

    'm

     [

    insert

     age

     or

     birth

     date

    ]

     years

     old

    ,

     and

     I

    'm

     currently

     [

    insert

     current

     location

    ].

     I

    'm

     [

    insert

     any

     hobbies

     or

     interests

     that

     I

     might

     have

    ].

     I

     love

     to

     [

    insert

     something

     that

     shows

     my

     personality

     or

     personality

     trait

    ],

     and

     I

    'm

     [

    insert

     something

     else

     that

     shows

     my

     personality

     or

     personality

     trait

    ].

     I

    'm

     a

     [

    insert

     any

     awards

     or

     recognition

     you

     might

     have

    ],

     and

     I

    'm

     [

    insert

     any

     other

     accol

    ades

     or

     honors

     you

     might

     have

    ].

     I

    'm

     [

    insert

     how

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     also

     the

     most

     populous

     city

     in

     the

     European

     Union

    .

     It

     is

     located

     on

     the

     island

     of

     France

     and

     has

     a

     population

     of

     approximately

     

    2

    7

     million

     people

    .

     Paris

     is

     the

     cultural

    ,

     economic

    ,

     and

     political

     center

     of

     France

     and

     the

     world

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     has

     been

     a

     major

     port

     and

     trade

     center

     for

     centuries

    .

     It

     is

     renowned

     for

     its

     architecture

    ,

     art

    ,

     and

     cuisine

    .

     Paris

     is

     also

     known

     for

     its

     world

    -ren

    owned

     fashion

     industry

    .

     The

     city

     has

     undergone

     significant

     urban

     renewal

     and

     has

     been

     transformed

     into

     a

     modern

     and

     vibrant

     met

    ropolis

    .

     Its

     iconic

     landmarks

    ,

     such

     as

     the

     E

    iff

    el

     Tower

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     complex

     and

     rapidly

     evolving

    ,

     driven

     by

     advances

     in

     technology

    ,

     changing

     societal

     norms

    ,

     and

     a

     growing

     emphasis

     on

     ethical

     and

     social

     responsibility

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

     that

     are

     currently

     under

     investigation

     and

     ongoing

     research

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     AI

     is

     increasingly

     being

     integrated

     into

     various

     industries

     to

     automate

     tasks

     and

     reduce

     human

     labor

    .

     This

     will

     lead

     to

     the

     widespread

     adoption

     of

     robotics

    ,

     drones

    ,

     and

     other

     automated

     systems

    ,

     which

     will

     likely

     result

     in

     increased

     efficiency

    ,

     cost

     savings

    ,

     and

     environmental

     benefits

    .
    


    2

    .

     Improved

     AI

     ethics

    :

     AI

     will

     continue

     to

     evolve

     and

     improve

     over

     time

    ,

     but

     there

     will

     likely

     be

     significant

     ethical

     concerns

     and

     challenges

     associated

    



```python
llm.shutdown()
```
