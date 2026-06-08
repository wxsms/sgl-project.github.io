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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.54it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  19%|█▉        | 11/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.74it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.93it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.16it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.25it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  14%|█▍        | 8/58 [00:00<00:01, 25.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  21%|██        | 12/58 [00:00<00:01, 29.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.44it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.44it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.05it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.05it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.05it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.05it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 36.05it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:00<00:00, 36.82it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 36.82it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.82it/s]

    Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 36.82it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.91it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.21it/s]

    Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.54it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 35.54it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.55it/s]

    Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.55it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  91%|█████████▏| 53/58 [00:01<00:00, 36.26it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 34.83it/s]


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
    Generated text:  Wanda, I am a mannequin artist, I graduated from the American College of Art and it is not my expertise to speak to, but there is an art of gifting handmade personalized items. I recently created a digital signature in the app that she titled "MemePin". She also added a meme for her new digital signature, which was a hot image of a dog walking on water. I like to make sure the meme is fun, amusing, and for some reason, gives a thumbs up to my gallery. Is the meme hot? The author is most likely to seek an opinion from someone familiar with art of gifting
    ===============================
    Prompt: The president of the United States is
    Generated text:  very popular. This makes him very good at having a lot of guests. He usually has about 700 people every year. The president likes to have a dinner party for guests. He likes to have a nice meal. He likes to make guests feel welcome. He likes to have a nice place to put the food and drinks. The president is very good at giving guests presents. He can buy presents for guests at home. He likes to have music. He likes to have some dancing. He likes to have a nice place to eat. The president likes to have a birthday party. He likes to have a New Year's party
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    Paris is a historical, cultural and artistic city located on the western bank of the Seine River, in the Île de la Cité, at the center of the metropolitan area of the Paris region. It is the capital of France. Paris is the largest city in the European Union and the fifth largest city in the world, and is the third largest city in France.
    Paris is the capital of France. It is located on the western bank of the Seine River, in the Île de la Cité, at the center of the metropolitan area of the Paris region. It is the capital of France. Paris is the
    ===============================
    Prompt: The future of AI is
    Generated text:  an exciting place. While AI is highly advanced, the rapid pace of progress means we need to be aware of the potential risks as well. In this guide, we will examine the possible risks of AI, discuss the benefits and potential of AI, and provide actionable steps to mitigate potential risks. To ensure that AI is used in a responsible and ethical way, it's important to consider the ethical implications of AI development and deployment. AI can be a powerful tool for creating innovation and improving the quality of life for everyone. By taking steps to mitigate potential risks and harnessing the benefits of AI, we can build a future where AI is a safe


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a few details about yourself, such as your age, gender, occupation, and any other relevant information]. I'm always looking for new opportunities and I'm always eager to learn new things. What do you do for a living? I'm always looking for new opportunities and I'm always eager to learn new things. What do you enjoy doing? I enjoy learning new things, trying new things, and meeting new people. What do
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that hosts the Eiffel Tower and is known for its rich history and culture. It is the largest city in France and is home to many of the country's most famous landmarks, including the Louvre Museum and Notre-Dame Cathedral. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many different cultures and ethnicities, and is a melting pot of different influences. It is a city that is constantly evolving and changing, and is a must-visit destination for anyone interested in French culture and history. 
    
    Paris is a city that is a true reflection of France's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with other technologies: AI is likely to become more integrated with other technologies such as machine learning, natural language processing, and computer vision. This integration will enable AI to perform tasks that are currently difficult or impossible for humans to accomplish.
    
    2. Enhanced privacy and security: As AI becomes more integrated with other technologies, there will be an increased need for privacy and security measures to protect the data and personal information that is generated and processed by AI systems.
    
    3. Greater emphasis on ethical considerations: As AI becomes more integrated with other technologies, there will be an increased need for
    


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
    Generated text:  [Your Name], and I am a [role] at [organization]. I am passionate about [one or two things you are passionate about] and [major interests or hobbies]. I have a [set of skills or abilities] that make me a great fit for this role and I am eager to learn more about [specific aspects of the organization or role]. How can I help you today?
    
    ---
    
    This introduction is neutral, open, and straightforward. It suggests that you're interested in the role, and you have a clear background or skill set that makes you a suitable candidate for the position. It also shows that you're eager to learn
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a city located in the heart of the French mainland and is the largest city in Europe and the second largest city in the world by population. The city is known for its historical landmarks, museums, and festivals. Paris is home to many of the world's leading arts institutions and is a major hub for cultural activities, including music, theater, and film. The city is also known for its elegant and refined atmosphere, with its many chateaux and high-end fashion and luxury shopping. Paris is the cultural and economic center of France and has a rich history and a unique blend of traditional and modern influences. The city has
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be driven by several key trends, including:
    
    1. Increased focus on ethical considerations: As AI becomes more widespread and complex, there will be increasing pressure to ensure that it is used ethically and responsibly. This will likely lead to further developments in ethical AI, such as the development of AI that is more transparent and accountable.
    
    2. Greater integration with human intelligence: As AI becomes more integrated with human intelligence, it is likely to be able to learn from human input and make decisions based on that input. This will likely create new opportunities for AI to contribute to society, such as in areas such as healthcare, transportation, and energy


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

     character

    's

     name

    ],

     and

     I

    'm

     a

     [

    insert

     the

     character

    's

     profession

     or

     nature

    ].

     I

     enjoy

     [

    insert

     any

     hobbies

     or

     interests

     that

     you

     enjoy

    ].


    I

    'm

     someone

     who

     thr

    ives

     in

     different

     environments

     and

     situations

    .

     I

    'm

     an

     [

    insert

     any

     attributes

     or

     skills

     that

     you

     consider

     essential

     for

     your

     job

    ].


    I

    'm

     always

     looking

     for

     ways

     to

     [

    insert

     any

     hobbies

     or

     areas

     of

     interest

     that

     you

    're

     interested

     in

    ].


    I

    'm

     a

     [

    insert

     any

     positive

     qualities

     that

     you

     find

     admirable

    ].

     I

    'm

     passionate

     about

     [

    insert

     any

     passions

     or

     hobbies

     that

     you

     enjoy

    ].


    I

    'm

     willing

     to

     [

    insert

     any

     willingness

     to

     learn

     or

     support

     positive

     change

     that

     you

     consider

     important

    ].


    Thank

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     elaborate

    :


    -

     Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     third

    -largest

     city

     in

     the

     European

     Union

    .


    -

     It

     was

     founded

     in

     

    7

    8

    9

     as

     the

     town

     of

     Paris

    .


    -

     It

     is

     the

     birth

    place

     of

     the

     French

     Revolution

     and

     the

     Romantic

     Movement

    .


    -

     Paris

     is

     renowned

     for

     its

     art

    ,

     culture

    ,

     and

     architecture

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .


    -

     It

     is

     a

     major

     transportation

     hub

     and

     is

     home

     to

     many

     of

     France

    's

     most

     famous

     landmarks

    ,

     including

     the

     Palace

     of

     Vers

    ailles

     and

     the

     Notre

    -D

    ame

     Cathedral

    .


    -

     Paris

     is

     often

     referred

     to

     as

     "

    the

     City

     of

     Light

    "

     due

     to

     its

     iconic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     dynamic

     and

     rapidly

     evolving

    ,

     with

     many

     possible

     trends

     shaping

     the

     technology

    's

     direction

    .

     Here

     are

     some

     of

     the

     key

     trends

     that

     may

     influence

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     data

     processing

     power

    :

     As

     data

     processing

     power

     continues

     to

     grow

    ,

     AI

     systems

     will

     become

     more

     powerful

     and

     capable

     of

     processing

     large

     amounts

     of

     data

     quickly

     and

     accurately

    .

     This

     will

     enable

     AI

     to

     better

     understand

     human

     behavior

     and

     make

     more

     intelligent

     decisions

    .
    


    2

    .

     More

     complex

     algorithms

    :

     AI

     will

     continue

     to

     become

     more

     complex

     and

     sophisticated

    ,

     allowing

     it

     to

     learn

     from

     a

     wider

     range

     of

     data

     and

     develop

     more

     nuanced

     understanding

     of

     human

     behavior

    .
    


    3

    .

     Greater

     integration

     with

     human

     intelligence

    :

     AI

     systems

     will

     become

     more

     integrated

    



```python
llm.shutdown()
```
