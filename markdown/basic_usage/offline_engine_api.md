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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:31,  4.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.09it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.09it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.69it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.36it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:05<00:00, 21.22it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.90it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.90it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 17.90it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:02, 18.46it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.17it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.86it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 37.29it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 40.25it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=352 avail_mem=75.74 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=320 avail_mem=75.73 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=288 avail_mem=75.71 GB):  53%|█████▎    | 31/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=288 avail_mem=75.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=256 avail_mem=75.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=240 avail_mem=75.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=224 avail_mem=75.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]

    Capturing num tokens (num_tokens=208 avail_mem=75.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=192 avail_mem=75.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 43.23it/s]Capturing num tokens (num_tokens=192 avail_mem=75.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=176 avail_mem=75.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=160 avail_mem=75.01 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=144 avail_mem=75.00 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=128 avail_mem=75.00 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  71%|███████   | 41/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=112 avail_mem=75.00 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=96 avail_mem=75.00 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s] Capturing num tokens (num_tokens=80 avail_mem=74.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=64 avail_mem=74.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=28 avail_mem=74.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=24 avail_mem=74.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=20 avail_mem=74.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=16 avail_mem=74.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=12 avail_mem=74.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.83it/s]Capturing num tokens (num_tokens=12 avail_mem=74.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=8 avail_mem=74.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.21it/s] Capturing num tokens (num_tokens=4 avail_mem=74.96 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.21it/s]Capturing num tokens (num_tokens=4 avail_mem=74.96 GB): 100%|██████████| 58/58 [00:01<00:00, 39.20it/s]


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
    Generated text:  Emily and I am a writer who has been writing since 2012. I have a passion for the arts, education, and connecting with others. I have a degree in creative writing from Florida State University.
    I am a freelance writer, speaker, and instructor. I have worked with schools, universities, and businesses as a writer. I have written for magazines, websites, and social media. I have over 75, 000 followers on Twitter, and I also have a Twitter handle that is used for writing workshops and guest lectures.
    Emily has a keen eye for detail and a strong sense of humor. She
    ===============================
    Prompt: The president of the United States is
    Generated text:  a high-ranking political office. It is a government position, and the president is the head of government. Which of the following options is a correct example of a high-ranking political office?
    
    A. Mayor of a city
    
    B. Council of States
    
    C. President of the United States
    
    D. Speaker of the House of Representatives
    
    E. Prefect of a province
    
    To determine which of the options is a correct example of a high-ranking political office, let's analyze each option step by step.
    
    1. **Mayor of a city**: This is a local government position, not a high-ranking political office. A mayor is a city
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Madrid
    D. Rome
    
    A. Paris. 
    
    Paris is the capital of France, located in the northwestern part of the country. It is known for its stunning architecture, rich history, and vibrant culture. Other options like London, Madrid, and Rome are not capitals of their respective countries. Madrid is the capital of Spain, Rome is the capital of Italy, and London is the capital of the United Kingdom. The United States does not have a capital city.
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but not everyone agrees. For many, the latest developments in AI are seen as a threat to privacy and data protection. How can we reconcile AI with data protection and privacy?
    
    Cybersecurity has become a priority for businesses, and AI has become a major new technology in that regard. AI applications, including automated decision-making, must be designed with data protection and privacy in mind, and businesses need to be aware of the risks that they face.
    
    This article explains what you should consider when it comes to AI and data protection and privacy. It will also discuss the risks and ways to mitigate them.
    
    What is AI and how does it


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] [Vehicle Name] and I'm currently [Current Location]. I'm [Your Name] and I'm a [Your Profession] [Your Job Title]. I'm [Your Age] years old and I'm [Your Gender]. I'm [Your Nationality] and I'm [Your Religion]. I'm [Your Ethnicity] and I'm [Your Language]. I'm [Your Favorite Color] and I'm [Your Favorite Animal]. I'm [Your Favorite Book] and I'm [Your Favorite Movie].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art, and cuisine. Paris is a cultural and economic hub of France and a major tourist destination. It is home to many world-renowned museums, theaters, and landmarks. The city is also known for its annual festivals and events, including the Eiffel Tower Festival and the World of Dancer Festival.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This integration will allow AI systems to learn from and interact with a wider range of data sources, leading to more accurate and effective predictions.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This will require the development of new algorithms and techniques to protect sensitive data and prevent unauthorized access.
    
    3. Greater automation and efficiency: AI is likely
    


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
    Generated text:  [Your Name] and I am an [Age] year old [Occupation]. I am a software engineer by profession, and I have been working in the industry for [X] years. I enjoy [one or two hobbies or interests] and I am always looking for ways to improve [any skill or area of expertise]. I am constantly learning and growing, and I am always striving to do my best. I am excited to talk to you! How are you? [You] [Your Name] [You] [You] [Your Name] [Your Name] [Your Name] [Your Name] [Your Name]
    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and is known for its historical architecture, museums, and artistic culture. Paris is also known as "La République" (the Republic) and is the birthplace of Napoleon Bonaparte. It is often called the "City of Light" due to its vibrant cultural scene and the city's skyline. Paris is a major transportation hub with many public transportation options and is home to many iconic landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is a popular tourist destination and attracts millions of visitors each year. The city's well-preserved
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly dynamic and unpredictable, and there are many possible trends that could shape the field in the coming years. Some of the most likely trends include:
    
    1. Increased automation and robotics: As AI technology continues to evolve, we can expect to see more robotic and automation-driven AI systems take over many tasks that previously required human intervention. This could lead to significant job losses in many industries, but also create new opportunities for job creators and retraining programs.
    
    2. Personalized AI: As AI technology improves, we may see the development of personalized AI systems that can learn from individual data to provide customized solutions and recommendations. This could lead to increased efficiency


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

    __

     and

     I

    'm

     a

    /an

     __

    ________

    _

    .


    I

     work

     at

     __

    ________

    __

     (

    location

    ).

     My

     job

     involves

     __

    ________

    __.

     I

    'm

     the

     best

     __

    ________

    _

     because

     I

     have

     __

    ________

    __.

     I

    'm

     also

     the

     __

    ________

    _

     of

     __

    ________

    __.

     I

    'm

     excited

     to

     meet

     you

    !

     How

     can

     I

     assist

     you

     today

    ?

     Let

     me

     know

     how

     I

     can

     help

     you

     learn

     more

     about

     me

     and

     what

     I

     can

     do

     for

     you

    !

     What

     do

     you

     want

     to

     know

     about

     me

    ?

     I

    'd

     be

     happy

     to

     hear

     about

     my

     background

    ,

     my

     interests

    ,

     my

     personality

    ,

     and

     why

     I

    'm

     the

     perfect

     fit

     for

     your

     job

    !

     How

     can

     I

     help

     you

     find

     out

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    To

     elaborate

     on

     the

     statement

    :

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     most

     populous

     metropolitan

     area

     in

     the

     world

    ,

     with

     an

     estimated

     population

     of

     over

     

    2

    .

    3

     million

     inhabitants

    .

     It

     is

     located

     in

     the

     Paris

     Region

     and

     is

     the

     administrative

     capital

     of

     the

     metropolitan

     region

    ,

     which

     includes

     the

     cities

     of

     Paris

    ,

     Mont

    mart

    re

    ,

     and

     the

     surrounding

     area

    .

     Paris

     is

     known

     for

     its

     iconic

     architecture

    ,

     rich

     culture

    ,

     and

     vibrant

     culture

     scene

    ,

     and

     is

     one

     of

     the

     world

    's

     most

     important

     cities

     for

     commerce

    ,

     education

    ,

     and

     tourism

    .

     It

     is

     also

     a

     symbol

     of

     French

     identity

     and

     a

     major

     tourist

     destination

    ,

     and

     is

     the

     world

    's

     largest

     city

     by

     population

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     evolving

    .

     Some

     of

     the

     most

     common

     trends

     are

    :
    


    1

    .

     Machine

     learning

     and

     deep

     learning

    :

     These

     are

     the

     two

     main

     approaches

     to

     machine

     learning

    .

     They

     are

     becoming

     increasingly

     sophisticated

    ,

     and

     have

     the

     potential

     to

     make

     incredible

     progress

     in

     various

     fields

    .
    


    2

    .

     Natural

     language

     processing

    :

     As

     the

     human

     language

     is

     incredibly

     complex

    ,

     it

     is

     becoming

     increasingly

     important

     to

     develop

     algorithms

     that

     can

     understand

     and

     generate

     natural

     language

    .

     This

     includes

     things

     like

     chat

    bots

    ,

     virtual

     assistants

    ,

     and

     virtual

     assistants

     for

     human

     language

    .
    


    3

    .

     Robotics

    :

     This

     is

     the

     field

     of

     artificial

     intelligence

     that

     is

     focused

     on

     building

     robots

     that

     can

     perform

     a

     wide

     range

     of

     tasks

    .

     These

     robots

     are

     being

     used

     in

    



```python
llm.shutdown()
```
