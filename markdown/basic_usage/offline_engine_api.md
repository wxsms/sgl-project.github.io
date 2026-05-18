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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.87it/s]


    2026-05-18 15:12:33,322 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 15:12:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:38,  3.84s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:31,  1.68it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  5.00it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 10.50it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 17.13it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 25.98it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 35.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.79it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.60 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:02, 19.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.59 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 22.45it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.39it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  21%|██        | 12/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  21%|██        | 12/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  21%|██        | 12/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  21%|██        | 12/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.96it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s] Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=832 avail_mem=72.53 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  34%|███▍      | 20/58 [00:00<00:01, 32.54it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]

    Capturing num tokens (num_tokens=640 avail_mem=72.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=576 avail_mem=72.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.32it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  52%|█████▏    | 30/58 [00:00<00:00, 40.18it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  52%|█████▏    | 30/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]

    Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=240 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 42.87it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 44.93it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s]

    Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 46.14it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=32 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.47it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.47it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 38.44it/s]


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
    Generated text:  John and I live in the UK. I am a business person who has been working in the UK for a long time now and I have been using the general data protection regulations (GDPR) in my work. However, I have just received a message from a customer to delete all my data from my company's website (which is located in Germany) and replace it with a competitor's website located in the UK. I am very worried because I have access to a lot of data (both sensitive and non-sensitive) on this website. Do I have any legal recourse to take action against the competitor if I do not delete my data
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. His or her job involves a lot of different things. They might have a very important job when it comes to the country. In the past, they might have been more or less important. However, now, the country is different. The president has become more of a helper to the country. It is not the President’s job to run the country, but to make sure that the country is safe. The president is also trying to make sure that America is a great place to live. The president is always busy and very busy. So, the president has to get along with everyone in the country. He or
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Rome
    C. London
    D. Tokyo
    Answer: A
    
    To ensure that the quality of the entire team's work is the best, the most appropriate strategy is to ____.
    A. Set a good example in working style and behavior
    B. Have all team members who are in charge of work to be proficient in all work processes
    C. Focus on personal growth and development of team members
    D. Enhance communication among team members
    Answer: A
    
    Mr. Li has a son named Xiao Ming and a daughter named Xiao Fang. Xiao Ming is 5 years old, and Xiao
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and we are all trying to figure out what the future holds for our humanity, our planet and our way of life. What if we could change the future of AI with simple code? Well, we can certainly change the future of AI with simple code.
    
    ## Why Code Should Be Advanced
    
    We are now entering the era of the cloud, the quantum and the era of AI. It is an era that will see an exponential growth of AI systems.
    
    So what will it be like for humans to interact with the world of AI? Will we need to learn to interact with this world? Will we need to learn to understand the systems that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character]. I enjoy [insert a short description of your character's interests or hobbies]. I'm always looking for new challenges and opportunities to grow and learn. What do you think makes you unique? I'm a [insert a short description of your character's personality or traits]. I'm always looking for ways to improve myself and make the world a better place. What's your favorite hobby or activity? I enjoy
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. Its rich history and diverse culture make it a fascinating city to explore and experience. 
    
    The city is also home to many notable French artists, writers, and musicians, including Pablo Picasso, André Breton, and Claude Monet. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI is likely to continue to be used for a wide range of applications, from healthcare and finance to transportation and entertainment, as it becomes more accessible and affordable. As AI becomes more integrated into our daily lives, it is likely to have a significant impact on our society and the way we live and work. However, it is also important to consider the potential risks and challenges
    


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
    Generated text:  [First Name] and I am [Last Name]. I am a [job title] with [number of years] years of experience in [industry]. I am currently a [position] at [company name]. In my free time, I enjoy [interest or hobby]. I am [age]. [First Name] and I am [Last Name]. I am a [job title] with [number of years] years of experience in [industry]. I am currently a [position] at [company name]. In my free time, I enjoy [interest or hobby]. I am [age]. [First Name] and I am [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical significance, art, architecture, fashion, and music. It is the largest city in Europe by population and the sixth-largest in the world by area. The city has a rich cultural heritage and is home to numerous museums, galleries, and landmarks. Paris is renowned for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also home to a diverse range of international cuisine, including French cuisine, Italian, and American. Paris is a cultural and historical center that attracts millions of visitors annually and is one of the most visited cities in the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be highly dynamic and unpredictable, with potential developments in several areas. Here are some of the possible trends that may shape the future of AI:
    
    1. Increased integration of AI into everyday life: AI is expected to become more integrated into our daily lives, from self-driving cars to virtual assistants that can understand and respond to our queries. This integration may lead to a more pervasive use of AI in our lives, with widespread adoption of AI technologies in various sectors such as healthcare, finance, transportation, and education.
    
    2. Personalized AI: As AI becomes more advanced, it is likely to become more personalized, with AI systems designed to


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

    ],

     and

     I

    'm

     a

     [

    occupation

    ].

     I

     enjoy

     exploring

     new

     places

     and

     meeting

     new

     people

    ,

     and

     I

    'm

     always

     looking

     for

     interesting

     opportunities

     to

     learn

     and

     grow

    .

     I

    'm

     patient

    ,

     organized

    ,

     and

     respectful

     of

     others

    '

     opinions

     and

     ideas

    .

     I

     enjoy

     having

     fun

     and

     doing

     things

     that

     challenge

     me

    ,

     and

     I

    'm

     always

     eager

     to

     try

     new

     things

    .

     I

    'm

     a

     team

     player

     and

     I

     thrive

     on

     collaboration

     and

     communication

    .

     I

     like

     to

     stay

     positive

     and

     find

     solutions

     to

     problems

    .

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     and

     be

     better

    .

     I

    'm

     excited

     to

     meet

     you

    !

     How

     do

     you

     want

     to

     be

     introduced

     to

     you

    ?

     Sure

    ,

     here

    's

     a

     neutral

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     Europe

     and

     the

     world

    ’s

     

    6

    th

     most

     populous

     city

    .
    


    Key

     points

    :
    


     

     *

     Paris

     is

     the

     largest

     city

     in

     Europe

     and

     the

     world

    's

     

    6

    th

     most

     populous

     city

    .


     

     *

     Paris

     is

     the

     capital

     of

     France

    ,

     the

     largest

     and

     most

     populous

     country

     in

     Europe

    .


     

     *

     It

     is

     the

     main

     city

     of

     France

     and

     the

     biggest

     city

     in

     the

     European

     Union

    .

     


     

     *

     It

     is

     known

     for

     its

     museums

    ,

     grand

     pal

    aces

    ,

     and

     fashion

    .


     

     *

     Paris

     is

     the

     seat

     of

     the

     government

    ,

     legislative

    ,

     executive

     and

     judicial

     branches

     of

     the

     French

     government

    .

     
    


    Its

     population

     is

     estimated

     at

     over

     

    2

    .

     

    1

     million

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     number

     of

     factors

    ,

     including

     advances

     in

     machine

     learning

     and

     deep

     learning

    ,

     increasing

     data

     availability

     and

     availability

    ,

     continued

     advancements

     in

     computing

     power

    ,

     and

     evolving

     societal

     norms

     and

     expectations

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     The

     development

     of

     AI

     systems

     that

     can

     perform

     tasks

     that

     require

     judgment

     or

     decision

    -making

    ,

     such

     as

     medical

     diagnosis

     or

     criminal

     justice

    ,

     could

     lead

     to

     increased

     scrutiny

     of

     AI

     systems

     and

     a

     focus

     on

     ethical

     considerations

    .

     This

     could

     involve

     more

     stringent

     regulations

     on

     AI

     development

     and

     testing

    ,

     as

     well

     as

     greater

     transparency

     and

     accountability

     in

     AI

     systems

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     has

     the

    



```python
llm.shutdown()
```
