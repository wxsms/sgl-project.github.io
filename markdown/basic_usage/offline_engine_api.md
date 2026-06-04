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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.79it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:37,  4.88s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.00it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.00it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]

    Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.02it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 13.20it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:01, 19.98it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=75.16 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.71 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 17.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:02, 20.51it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.51it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.51it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.51it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.58it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.66 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 33.48it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.65 GB):  31%|███       | 18/58 [00:00<00:01, 33.48it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 33.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  31%|███       | 18/58 [00:00<00:01, 33.48it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  31%|███       | 18/58 [00:00<00:01, 33.48it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=896 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=704 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.84it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.61 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.88it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=384 avail_mem=74.62 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.80it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.80it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=240 avail_mem=74.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.83it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 43.32it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  81%|████████  | 47/58 [00:01<00:00, 40.85it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 42.17it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  98%|█████████▊| 57/58 [00:01<00:00, 43.29it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 37.65it/s]


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
    Generated text:  Christian. I am a self-taught programmer who loves to explore the world of coding and learning new skills. I am currently working as a software developer at a company called Tinkercast, where I am learning and contributing to a wide range of projects. My interests include programming, game development, and virtual reality.
    Could you please share any recent projects or programming languages that you have contributed to or worked on in the past? As a self-taught programmer, I have no formal education in programming. However, I have developed my skills in various programming languages, including Python, JavaScript, and Ruby. I have also worked on
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing a new policy. There are two possible scenarios: Scenario A and Scenario B. In Scenario A, the president plans to cut the budget for the Department of Health and Human Services. In Scenario B, the president plans to increase the budget for the Department of Health and Human Services. Each scenario will have an equal probability of occurring. What is the probability that the policy will either succeed or fail?
    To determine the probability that the policy will either succeed or fail, we need to consider the following:
    
    1. The policy can either succeed or fail, but we are not given any information about the success or failure of the policy. Therefore,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    答案:  A
    
    The train has stopped at the station because there is no signal.
    A. 正确
    B. 错误
    答案:  A
    
    2000年9月1日，甲公司从银行贷款1000万元，期限为2年，年利率为8%。2001年12月31日，甲公司应计提的预计负债为____万元。
    A. 10
    B. 12
    C. 20
    D. 24
    
    ===============================
    Prompt: The future of AI is
    Generated text:  intertwined with other fields, including medicine, education, and entertainment. AI technologies can improve our understanding of complex systems, enable the development of new treatments, and enhance educational outcomes. However, these technologies also raise concerns about privacy, bias, and the potential for misuse. What steps can be taken to ensure that these technologies are used ethically and responsibly?
    Certainly! Here are some key considerations for ensuring the ethical and responsible use of AI in various fields:
    1. Develop clear ethical guidelines for AI technologies: AI technologies should be developed with ethical considerations in mind, including issues such as bias, privacy, and transparency.
    2. Provide transparency in AI


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a unique trait or skill that makes you stand out]. And what's your background? I have a [insert a relevant background or education]. And what's your favorite hobby or activity? I enjoy [insert a hobby or activity that you enjoy]. And what's your favorite book or movie? I love [insert a favorite book or movie]. And what's your favorite place to go? I love [insert a favorite place]. And what
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city with a rich history and a diverse population. It is the largest city in France and the third-largest city in the world by population. Paris is famous for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also known for its fashion industry, art scene, and its role in the French Revolution. Paris is a cultural and economic hub of the country and is a popular tourist destination. The city is home to many famous museums, theaters, and restaurants, and is a major center for politics, business, and science
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between machines and humans. This could lead to more sophisticated forms of AI, such as those that can understand and adapt to human emotions and behaviors.
    
    2. Enhanced privacy and security: As AI becomes more integrated with human intelligence, there will be a greater need for privacy and security measures to protect against potential misuse of AI systems. This could lead to more stringent regulations and standards for AI development and deployment.
    
    3. Greater focus on ethical considerations: As AI becomes more integrated
    


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
    Generated text:  [Name], and I am [age] years old. I currently work in [your profession], and I enjoy [your profession's passion or hobbies]. I am always up-to-date on the latest trends and technologies, and I am a strong learner who loves to keep my skills sharp by taking on new challenges. I am always looking for the next great adventure, whether it be a new project or a new adventure. I am a friendly and approachable person who always looks for opportunities to contribute to the community and help others. I believe in the value of perseverance and the power of perseverance, and I strive to stay positive and upbeat at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in France and is the cultural and political center of the country. Its skyline is composed of many tall buildings, including the Eiffel Tower. It is known for its iconic landmarks, including Notre-Dame Cathedral, the Louvre Museum, and the Palace of Versailles. Paris is also famous for its rich history, including the role of the French Revolution and its artistic and cultural traditions. It is a world-renowned tourist destination and a major economic center in Europe. The city has a rich and diverse cultural scene, with various art forms, music, and cuisine. Paris is also a cultural hub
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and is expected to continue to evolve rapidly. Some possible future trends in artificial intelligence include:
    
    1. Increased reliance on AI for decision-making: As AI becomes more integrated into our daily lives, we are likely to see a greater emphasis on its use for decision-making processes, such as healthcare, finance, and transportation.
    
    2. Integration with human emotions and cognitive abilities: As AI systems become more sophisticated, they are likely to begin to incorporate more human emotions and cognitive abilities. This could include the ability to understand and respond to human emotions, as well as to develop new cognitive abilities.
    
    3. Enhanced privacy and data protection: As AI systems


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

    ]

     and

     I

    'm

     a

     [

    insert

     occupation

    ]

     who

     has

     always

     been

     fascinated

     by

     the

     mysteries

     of

     the

     universe

    .

     With

     a

     keen

     eye

     for

     detail

     and

     a

     drive

     to

     explore

     new

     ideas

    ,

     I

    'm

     always

     on

     the

     lookout

     for

     ways

     to

     expand

     my

     knowledge

     and

     broaden

     my

     hor

    izons

    .

     I

     believe

     that

     curiosity

     and

     an

     open

     mind

     are

     the

     keys

     to

     unlocking

     the

     secrets

     of

     the

     cosmos

    ,

     and

     I

    'm

     here

     to

     share

     my

     discoveries

     with

     you

    .

     Thanks

     for

     taking

     the

     time

     to

     learn

     about

     me

    .

     What

     is

     your

     name

     and

     what

     do

     you

     do

     for

     a

     living

    ?

     [

    insert

     name

    ]

     is

     a

     [

    insert

     occupation

    ]

     who

     has

     always

     been

     fascinated

     by

     the

     mysteries

     of

    
    
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

     city

     with

     a

     rich

     history

     and

     cultural

     significance

    ,

     known

     for

     its

     towering

     architecture

    ,

     vibrant

     arts

     and

     music

     scene

    ,

     and

     rich

     history

     dating

     back

     to

     ancient

     times

    .

     Paris

     is

     a

     major

     transportation

     hub

    ,

     with

     an

     extensive

     network

     of

     highways

    ,

     train

     stations

    ,

     and

     airports

    ,

     making

     it

     a

     highly

     diverse

     and

     bustling

     city

    .

     It

     is

     also

     a

     UNESCO

     World

     Heritage

     site

    ,

     recognized

     for

     its

     unique

     blend

     of

     history

    ,

     art

    ,

     and

     culture

    .

     The

     city

     is

     known

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

     Lou

    vre

     Museum

    ,

     and

     Notre

     Dame

     Cathedral

    ,

     and

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

    ,

     with

     potential

     impacts

     on

     various

     industries

     and

     areas

     of

     life

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     shape

     the

     way

     we

     live

     and

     work

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     automation

     and

     artificial

     intelligence

    :

     As

     automation

     and

     AI

     continue

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     and

     more

     jobs

     automated

    ,

     including

     manufacturing

    ,

     customer

     service

    ,

     and

     administrative

     roles

    .

     Additionally

    ,

     AI

     will

     be

     used

     in

     various

     industries

     to

     perform

     tasks

     that

     are

     currently

     done

     by

     humans

    ,

     such

     as

     image

     recognition

    ,

     speech

     recognition

    ,

     and

     predictive

     analytics

    .
    


    2

    .

     Improved

     decision

    -making

     and

     algorithm

    ic

     bias

    :

     As

     AI

     models

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     an

     increased

     need

    



```python
llm.shutdown()
```
