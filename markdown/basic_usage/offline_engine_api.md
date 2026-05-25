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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.99it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:04,  4.28s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.51it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.50it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.64it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 23.80it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 32.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.70it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.72it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.72 GB):  21%|██        | 12/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.71 GB):  21%|██        | 12/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.22 GB):  21%|██        | 12/58 [00:00<00:01, 24.92it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.22 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.21 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.09 GB):  28%|██▊       | 16/58 [00:00<00:01, 27.72it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.09 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.98it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.49it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 41.44it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 41.44it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.05it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s]

    Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.53it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.97it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.97it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.97it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.37it/s]


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
    Generated text:  Tuan. I am a 15-year-old girl from Hong Kong. I have a good friend named Xiaozhang. We are both very good students. We like sports. We like to eat dumplings. And we are good at singing and dancing. We also like to play the piano. We often go out for breakfast together. Our favorite food is a cup of milk. It is also the best place for us to relax. We like playing computer games. We always play for a long time when we play computer games. My favorite sport is basketball. I think playing basketball is very interesting. Playing basketball is fun, but
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader in the United States. He is the head of the executive branch and serves as the commander-in-chief of the armed forces of the United States. The president is the chief executive of the government of the United States, making decisions on important policy matters. The president is elected to a two-year term. He can be re-elected. He is usually not re-elected for consecutive terms. He is usually required to submit a biennial report to Congress.
    Does this next sentence follow, given the above sentence. "The President of the United States does not have the power to veto legislation."?
    Available options:
     a). yes
     b
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the most important city in Europe. It is the capital of France and the largest city in France. It is in the centre of the country. The city has a population of over 2 million people. The city is very popular with tourists and visitors from all over the world. The city is also very famous for its many beautiful churches and monuments. There are many tall buildings in Paris. There is a high speed train system that makes it easy for people to get around the city. Most of Paris has a population of over 2 million. The population is spread out over an area of about 90 square kilometres
    ===============================
    Prompt: The future of AI is
    Generated text:  what it is now. The emerging field of robotics is building models that can sense and move autonomously. What do these machines think? And do they actually act in their own interests? This paper offers a short introduction to robotic vision and modeling, while also considering the design of sensors and algorithms. The focus of this paper is on the environment and its interaction with a sensor, but the paper also discusses vision and robotics from a conceptual and historical perspective, with a focus on the development of the first robotics robots.
    Robotics is a field that has been around for over 100 years and, since its inception, has evolved and been


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


    Generated text:  Paris, also known as "La Ville Flottante" (floating city). It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also home to many world-renowned museums, theaters, and restaurants. Paris is a cultural and political center of France and a major tourist destination. The city is known for its rich history, art, and cuisine. It is also home to many international organizations and institutions. Paris is a city of contrasts,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and use, as well
    


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
    Generated text:  [Name], and I come from [Country]. I'm a [职业] with [Education] from [University]. I've [Number of years working] years of experience in [industry]. I love [job title] because [reason]. And I'm always [ability] to [action].
    Who is your favorite movie? What is it? What does it mean to you? In my opinion, "Pleasantville" is a funny, spontaneous, and fun movie. I mean, it's silly, it's funny, it's unexpected, and it's so fast-paced. To me, it's a reflection of my
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as "the City of Light" and the "City of Love".
    Paris is known for its iconic Notre-Dame Cathedral, lively nightlife, and diverse cultural scene. The city is also famous for its Opera Garnier and its romantic Parisian landscape. It has a population of around 2. 5 million people and is the 2nd most populous city in the world. Paris is a major cultural center with many renowned universities and museums. It is also home to the world's most famous landmark, the Eiffel Tower. Despite its many tourist attractions, Paris remains a popular tourist destination. Visitors can enjoy
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by several trends:
    
    1. Increased integration with human consciousness: As AI becomes more integrated with human consciousness, it will be possible to create AI that is more conscious and capable of self-awareness. This could lead to the development of more sophisticated AI systems that are able to perceive and understand the world around them, as well as communicate with humans in meaningful ways.
    
    2. Improved ethical and moral considerations: As AI systems become more advanced, they will be subjected to increased ethical and moral scrutiny. This could lead to a more rigorous and stringent set of guidelines and regulations for developing and deploying AI systems.
    
    3. Increased reliance on


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

    ],

     and

     I

    'm

     a

     [

    Your

     Profession

    ]

     with

     a

     [

    Your

     Education

    ]

     degree

     in

     [

    Your

     Major

    ].

     I

    'm

     passionate

     about

     [

    Your

     Passion

    /

    Interest

    /

    Job

    ]

     and

     I

    'm

     always

     up

     for

     learning

     new

     things

     and

     exploring

     new

     possibilities

    .

     I

    'm

     a

     quick

     learner

    ,

     enjoy

     challenging

     myself

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     grow

     and

     improve

    .

     I

    'm

     not

     afraid

     to

     take

     risks

     and

     be

     creative

    ,

     and

     I

    'm

     always

     eager

     to

     push

     the

     boundaries

     of

     what

    's

     possible

    .

     My

     strengths

     are

     my

     ability

     to

     adapt

     to

     new

     situations

    ,

     my

     curiosity

    ,

     my

     passion

     for

     learning

    ,

     and

     my

     desire

     to

     create

     meaningful

     connections

     with

     others

    .

     I

    'm

    
    
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

     home

     to

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

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     famous

     landmarks

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

     art

    ,

     and

     cuisine

    ,

     and

     has

     been

     a

     major

     center

     of

     trade

     and

     culture

     for

     over

     

    5

    0

    0

     years

    .

     Paris

     is

     an

     international

     city

     with

     many

     different

     cultures

     and

     languages

    ,

     and

     is

     home

     to

     many

     of

     the

     world

    's

     wealthiest

     and

     most

     influential

     people

    .

     According

     to

     a

     

    2

    0

    1

    7

     Forbes

     report

    ,

     Paris

     is

     the

     world

    's

     

    7

    th

     most

     expensive

     city

    .

     Despite

     its

     reputation

     for

     luxury

    ,

     Paris

     is

     also

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     full

     of

     exciting

     developments

     and

     breakthrough

    s

    .

     Here

     are

     some

     of

     the

     possible

     future

     trends

     that

     we

     can

     expect

     in

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     are

     expected

     to

     become

     more

     precise

     and

     accurate

    .

     This

     means

     that

     we

     can

     expect

     to

     see

     more

     and

     better

     results

     from

     AI

    -driven

     technologies

     in

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     manufacturing

    .
    


    2

    .

     Integration

     of

     AI

     with

     other

     technologies

    :

     As

     AI

     continues

     to

     evolve

    ,

     we

     can

     expect

     that

     it

     will

     become

     more

     integrated

     with

     other

     technologies

     such

     as

     IoT

    ,

     blockchain

    ,

     and

     deep

     learning

    .

     This

     integration

     will

     allow

     AI

     to

     perform

     tasks

     that

     were

     previously

     only

     possible

     with

    



```python
llm.shutdown()
```
