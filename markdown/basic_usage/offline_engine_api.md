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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.42it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.73it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.73it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:04,  7.82it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.94it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 18.98it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 24.84it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 33.84it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:03, 18.20it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 21.13it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.61it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.61it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.61it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  21%|██        | 12/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.12 GB):  21%|██        | 12/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.11 GB):  21%|██        | 12/58 [00:00<00:01, 29.37it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.11 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.27it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.10 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.08 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.01 GB):  26%|██▌       | 15/58 [00:00<00:01, 25.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.01 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=73.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s]Capturing num tokens (num_tokens=960 avail_mem=73.44 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s] Capturing num tokens (num_tokens=896 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s]Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  33%|███▎      | 19/58 [00:00<00:01, 28.06it/s]

    Capturing num tokens (num_tokens=832 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=768 avail_mem=73.43 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=704 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=640 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=576 avail_mem=73.42 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=512 avail_mem=73.40 GB):  50%|█████     | 29/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=480 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=448 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=416 avail_mem=73.42 GB):  50%|█████     | 29/58 [00:00<00:00, 36.56it/s]Capturing num tokens (num_tokens=384 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:01<00:00, 36.56it/s]Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  50%|█████     | 29/58 [00:01<00:00, 36.56it/s]

    Capturing num tokens (num_tokens=352 avail_mem=73.41 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=320 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=288 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=256 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=240 avail_mem=73.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  59%|█████▊    | 34/58 [00:01<00:00, 38.96it/s]Capturing num tokens (num_tokens=224 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=208 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=192 avail_mem=73.39 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=176 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=160 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.15it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=128 avail_mem=73.38 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=112 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=96 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s] Capturing num tokens (num_tokens=80 avail_mem=73.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.84it/s]Capturing num tokens (num_tokens=64 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=48 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=32 avail_mem=73.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=28 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=24 avail_mem=73.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]

    Capturing num tokens (num_tokens=20 avail_mem=73.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=20 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=16 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=12 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=8 avail_mem=73.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.29it/s] Capturing num tokens (num_tokens=4 avail_mem=73.33 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.29it/s]Capturing num tokens (num_tokens=4 avail_mem=73.33 GB): 100%|██████████| 58/58 [00:01<00:00, 34.88it/s]


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
    Generated text:  Bob. I'm 13 years old. I love to play sports. I like to play football. My favorite team is the Leeds United Football Club. They play in the English Premier League. I love to play football because it is very fun and I like to play with my friends. My favorite player is Michael Owen. He is my favorite player because I like to watch him play football. He is a big player with a lot of strength. I also like to play rugby. I like to watch it on TV. My favorite player is Billy Vunipola. He is my favorite player because I like to watch him play
    ===============================
    Prompt: The president of the United States is
    Generated text:  proposing to give $10,000,000 to a charity, to be used for education. The president needs to send a check for this amount to the charity by the end of the year, but the president needs to also budget $5,000,000 for other expenses. If the president decides to send the check for a quarter (6 months) and plans to use the remaining funds for education, how much money will be left in the total amount to be spent by the end of the year? To determine how much money will be left in the total amount to be spent by the end
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is situated in the center of the Île-de-France region and is the capital of the Languedoc-Roussillon department, as well as the second-largest city in France, after Paris. The city is located at the eastern end of the coast of the Mediterranean Sea, north of the Pyrenees. It has a population of 1.18 million people, and is the seventh-largest city in the European Union. The city is the seat of the French Government, and the capital of the French Republic.
    What is the capital of France? Paris. Its capital is the city of Paris, located
    ===============================
    Prompt: The future of AI is
    Generated text:  approaching, but how will it shape our lives? Join us for the final panel in the AI42 series, where leading experts from the industry will discuss how AI is changing the way we work, live, and make decisions in the future.
    The panel includes four leading figures from the tech industry – including:
    • Microsoft CEO Satya Nadella
    • Apple CEO Tim Cook
    • Google CEO Sundar Pichai
    • IBM CEO Andy Sun
    What can we expect from AI in 2023? Let's hear from the experts.
    This event is co-hosted by the Future of Jobs initiative at Silicon Valley University and


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich history and culture, and is a popular tourist destination. It is the largest city in France and one of the most visited cities in the world. The city is known for its fashion industry, art scene, and its role in the French Revolution. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of people, culture, and history
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial intelligence: As AI continues to advance, we can expect to see more automation and AI-driven technologies being developed and implemented in various industries. This could lead to increased efficiency, productivity, and cost savings for businesses.
    
    2. Enhanced privacy and security: As AI becomes more integrated into our daily lives, there will be an increased need for privacy and security measures to protect personal data. This could lead to the development of new technologies and protocols to ensure that AI
    


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
    Generated text:  [Name] and I'm a [Age] year old. I'm [Occupation] [or] a [Other Occupation] [or] [Other Occupation]. I'm a [Other Occupation] [or] a [Other Occupation], [Age] years old, and I live in [Your Location]. I'm passionate about [My Passion] and I'm always looking for [My Future Goal or Achievements]. I'm a [My Profession] [or] a [My Profession] [or] [My Profession], [My Profession]. My [My Profession] [or] a [My Profession] [or] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Paroi des Dames."
    Paris is a bustling and popular city in Western Europe, famous for its historic landmarks such as Notre Dame Cathedral, the Eiffel Tower, and the Louvre Museum. It's also home to the Louvre Museum, a world-renowned collection of art, and the Champs-Élysées, a famous avenue with numerous shops, restaurants, and theaters. Paris is known for its rich history, diverse culture, and vibrant nightlife, making it an iconic city of France. Paris, also known as "la Paroi des Dames," is home to many notable events
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be dominated by a combination of trends such as:
    
    1. Increased automation: With the rise of automation, AI will become more prevalent in everyday tasks and will be integrated into various industries. This will lead to the automation of repetitive and mundane tasks, freeing up more human workers for more creative and high-level work.
    
    2. Improved natural language processing: As AI technology advances, it will become better at understanding and generating natural language. This will enable more sophisticated language and text processing, leading to a wider range of applications such as chatbots, voice assistants, and more sophisticated forms of language-based communication.
    
    3. Enhanced cognitive abilities: AI


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

     professional

     [

    Job

     Title

    ].

     I

     am

     passionate

     about

     [

     passion

     or

     area

     of

     expertise

     ]

     and

     [

     What

     you

    're

     good

     at

     ]

     and

     I

     work

     hard

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     always

     looking

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     excited

     to

     work

     with

     anyone

     who

     shares

     my

     vision

     for

     the

     future

    .

     

    🌟

    ✨

    
    


    That

    's

     a

     great

     self

    -int

    roduction

    !

     Can

     you

     give

     me

     some

     examples

     of

     areas

     of

     expertise

     or

     passions

     you

     have

    ?

     Also

    ,

     do

     you

     have

     any

     hobbies

     or

     interests

     besides

     work

    ?

     

    🕒

    ✨

    
    


    Absolutely

    !

     In

     terms

     of

     areas

     of

     expertise

    ,

     I

     have

     a

     deep

     understanding

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     known

     for

     its

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .
    


    That

    's

     great

    !

     Can

     you

     tell

     me

     more

     about

     the

     history

     of

     Paris

     and

     its

     significance

     in

     French

     culture

    ?

     Certainly

    !

     Paris

     has

     a

     rich

     history

     dating

     back

     centuries

    ,

     with

     its

     origins

     dating

     back

     to

     ancient

     times

    .

     The

     Romans

     founded

     the

     city

     of

     Car

    th

    age

     in

     the

     

    2

    nd

     century

     AD

    ,

     and

     it

     became

     a

     major

     trading

     hub

     and

     capital

     of

     the

     Roman

     Empire

    .

     In

     the

     

    5

    th

     century

     AD

    ,

     the

     city

     was

     Christian

    ized

     by

     the

     emperor

     Constant

    ine

     I

    ,

     leading

     to

     the

     formation

     of

     the

     Latin

    -speaking

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     varied

    ,

     and

     there

     are

     many

     possible

     trends

     that

     could

     shape

     the

     development

     of

     the

     technology

    .

     Here

     are

     some

     potential

     trends

     to

     expect

    :
    


    1

    .

     Autonomous

     vehicles

    :

     With

     the

     rapid

     growth

     of

     autonomous

     vehicle

     technology

    ,

     there

     will

     likely

     be

     a

     wide

     range

     of

     companies

     developing

     vehicles

     that

     can

     drive

     themselves

     or

     assist

     humans

     in

     driving

    .

     This

     could

     include

     self

    -driving

     cars

     that

     can

     operate

     on

     public

     roads

    ,

     self

    -driving

     trucks

    ,

     and

     even

     planes

    .
    


    2

    .

     Personal

    ized

     medicine

    :

     AI

     will

     be

     used

     to

     analyze

     vast

     amounts

     of

     medical

     data

    ,

     including

     genetic

     information

    ,

     to

     help

     doctors

     develop

     personalized

     treatment

     plans

     for

     patients

    .

     This

     could

     lead

     to

     more

     effective

     and

     personalized

     treatments

     for

     diseases

     like

     cancer

    



```python
llm.shutdown()
```
