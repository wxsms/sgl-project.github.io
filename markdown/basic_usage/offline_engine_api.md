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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:04<00:46,  1.17it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:04<00:12,  3.71it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  9.24it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.38it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 23.50it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 32.60it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 32.60it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 32.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 18.74it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.73it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 29.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s]

    Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.63it/s] Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]

    Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  47%|████▋     | 27/58 [00:00<00:00, 40.10it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=288 avail_mem=74.08 GB):  55%|█████▌    | 32/58 [00:00<00:00, 39.47it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  55%|█████▌    | 32/58 [00:01<00:00, 39.47it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=224 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]

    Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.74it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=160 avail_mem=74.06 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.68it/s] Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  81%|████████  | 47/58 [00:01<00:00, 41.84it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.15it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=4 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.11it/s]Capturing num tokens (num_tokens=4 avail_mem=74.01 GB): 100%|██████████| 58/58 [00:01<00:00, 37.65it/s]


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
    Generated text:  Zenn and I am a computer program. I am here to help you solve problems and to answer questions to the best of my ability. Is there anything specific you would like to ask me? 
    
    There are many computer programs available that can help with your problem. Some programs can be helpful for learning programming, while others can be more useful for specific tasks. Is there anything in particular you would like to try or learn? 
    
    I'm here to help, and I'm always here to assist you. How can I assist you today? 
    
    [END_OF_TEXT] 
    
    Using the given text, write a Python script that asks the user
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. He is the leader of the government. The president is the head of the executive branch. The executive branch is the branch of the government responsible for carrying out the policies of the congress (legislative branch). He or she will be responsible for making decisions and the laws of the country. The president has a lot of responsibilities. He or she has to make decisions quickly. He or she has to lead the country. But he or she has to be careful not to take any advantage of the situation. What do you think the president should do? He or she has to be careful not to take any advantage of the
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Paris
    C. Paris
    D. Paris
    Answer:
    A
    
    According to the 'Construction Law of the People's Republic of China', which of the following statements about the issuance of construction permits is correct? 
    A. The approval of a construction permit shall be subject to inspection. 
    B. The validity period of a construction permit does not exceed 2 years. 
    C. The administrative agency issuing a construction permit shall review the safety facilities of the construction project. 
    D. The administrative agency issuing a construction permit shall organize the review of the safety facilities of the construction project. 
    Answer:
    
    ===============================
    Prompt: The future of AI is
    Generated text:  far from over
    
    An AI chatbot at a bank's digital branch
    
    The future of AI is far from over
    
    An AI chatbot at a bank's digital branch
    
    AI is changing the way we live. But is it also changing the way we think? A recent study suggests so.
    
    A team from the University of Oxford's Department of Statistics have discovered that in addition to being able to perform tasks more efficiently, AI is able to understand and explain the human mind more than any other technology in the world.
    
    The study, led by Professor Ian Goodfellow, a computer science researcher at the University of Oxford, shows that AI can


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics in Europe. Paris is a bustling metropolis with a rich history dating back to the Roman Empire and the Middle Ages. The city is home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. It is also home to many notable restaurants, including the famous Eiffel Tower restaurant. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Here are some possible future trends in AI:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the impact of AI on society.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This will require advances in areas such as machine learning, natural language processing, and computer vision.
    
    3.
    


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
    Generated text:  [Name], and I am a [profession] who has been [career goal] for [number] years. I strive to [character trait] and I am always looking for opportunities to [action] that will help me achieve my goals. What brings you here today? Welcome to [Name], I am here to provide you with all the information you need to make the most of your career, and I am here to help you grow and succeed in your endeavors! Let's connect. [Name] [Phone Number] [Email Address] [Address] [City, State, ZIP Code] [LinkedIn Profile] [Twitter Profile]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the heart of the French Riviera. 
    
    Step 1: Identify the capital city
    The capital city of France is Paris.
    
    Step 2: Gather information about the capital city
    - Paris is located in the heart of the French Riviera
    - It is the capital of France
    - Paris is known for its rich history, culture, and beautiful architecture
    - It is a popular tourist destination and is known for its fashion, food, and art scenes
    - Paris is home to the Eiffel Tower and the Louvre Museum
    
    Step 3: Craft a concise factual statement about Paris
    "Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, and the path forward is filled with possibilities. Here are some potential trends in AI that are likely to shape the next decade:
    
    1. Increased integration with human intelligence: As AI becomes more integrated with human intelligence, it may become more natural and intuitive. This could lead to new ways of interacting with AI systems, such as more natural language processing and better understanding of emotional intelligence.
    
    2. Ethical concerns: As AI becomes more integrated with human intelligence, there are ethical concerns about privacy, bias, and the potential for misuse. There will need to be a more nuanced approach to AI development, with more attention paid to ethical considerations.
    
    


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

    Position

    ]

     at

     [

    Company

    ].

     I

    'm

     a

     [

    Short

     Biography

    ]

     of

     [

    Name

    ]

     who

     specialize

     in

     [

    Role

    ].

     Welcome

     to

     [

    Company

    ],

     my

     personal

     team

     will

     assist

     you

     in

     [

    Purpose

     or

     Goal

    ].

     What

     can

     I

     do

     for

     you

     today

    ?

     What

     is

     your

     current

     task

     or

     project

    ?

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

     and

     I

     will

     be

     here

     to

     help

    !

     [

    Name

    ].

     [

    Position

    ]

     at

     [

    Company

    ]

     is

     a

     full

    -stack

     software

     engineer

     with

     a

     passion

     for

     [

    Company

    's

     mission

    ].

     My

     background

     in

     [

    relevant

     coursework

     or

     industry

     experience

    ]

     is

     an

     asset

     to

     [

    Company

    's

     mission

    ].

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     has

     been

     the

     city

     of

     power

     and

     culture

     for

     more

     than

     

    3

    0

    0

     years

    .

     Paris

     has

     a

     rich

     history

     and

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

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     cuisine

     and

     fashion

    ,

     with

     Paris

    ian

     cuisine

    ,

     fashion

    ,

     and

     art

     being

     a

     major

     draw

     for

     tourists

    .

     Paris

     is

     a

     bustling

     city

     with

     a

     strong

     sense

     of

     community

     and

     has

     become

     an

     iconic

     city

     in

     French

     culture

     and

     beyond

    .

     The

     city

     is

     home

     to

     many

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    ,

     and

     its

     skyline

     is

     dotted

     with

     high

    -rise

     buildings

    .

     Overall

    ,

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     an

     increase

     in

     automation

    ,

     self

    -learning

    ,

     and

     integration

     with

     natural

     language

     processing

    .

     The

     development

     of

     AI

     technologies

     is

     expected

     to

     continue

     to

     advance

     rapidly

    ,

     with

     new

     features

     and

     improvements

     being

     developed

     every

     day

    .

     The

     benefits

     of

     AI

    ,

     such

     as

     increased

     efficiency

    ,

     accuracy

    ,

     and

     speed

    ,

     are

     expected

     to

     continue

     to

     grow

    ,

     while

     the

     potential

     risks

     and

     challenges

     may

     also

     increase

    .

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     it

     is

     likely

     to

     have

     a

     significant

     impact

     on

     society

    ,

     economy

    ,

     and

     culture

    .

     Ultimately

    ,

     the

     future

     of

     AI

     is

     uncertain

    ,

     and

     its

     impact

     will

     depend

     on

     a

     variety

     of

     factors

    ,

     including

     how

     it

     is

     developed

    ,

     implemented

    



```python
llm.shutdown()
```
