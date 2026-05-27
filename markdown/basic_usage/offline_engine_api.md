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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:11,  4.42s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.47it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.37it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]

    Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.19it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:02, 12.93it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:01, 17.43it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 17.43it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 17.43it/s]

    Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 17.43it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:05<00:01, 17.43it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:05<00:01, 17.43it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 26.12it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 35.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.79it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.48it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 32.48it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.45it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.45it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.45it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  41%|████▏     | 24/58 [00:00<00:01, 25.62it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 25.62it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 25.62it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 25.62it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  41%|████▏     | 24/58 [00:00<00:01, 25.62it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]

    Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:01<00:01, 28.40it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.02it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.02it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.02it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.02it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.02it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 30.59it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 33.66it/s] Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=48 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  81%|████████  | 47/58 [00:01<00:00, 36.04it/s]Capturing num tokens (num_tokens=28 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=24 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]Capturing num tokens (num_tokens=12 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 38.79it/s] Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  98%|█████████▊| 57/58 [00:01<00:00, 41.65it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 33.13it/s]


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
    Generated text:  Naiomi and I am a writer, poet, and award-winning artist. I am from Grand Prairie, Texas. I grew up in the small town of Fort Worth, Texas where I attended the University of Texas at Austin. I lived in Los Angeles and Houston for several years while I was working in the business and arts industry. I have always been interested in making art and writing and I am inspired by the work of many artists and writers. I have been a long-time member of the city of Fort Worth Arts Alliance and I enjoy helping to support and create the art and culture in our community. I have worked as a graphic designer
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president has many important jobs. His job is to represent the country. The president has to do many important things. Here are some important things the president has to do. The president has to help the country. He has to make sure that all the people in the country are safe. He also has to make sure that the people in the country are happy. The president has to try to solve problems in the country. He has to try to help the people who are having problems. The president has to work with the people who are in charge of the country. He has to try to make sure that all
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The official language is French. You can eat a pluie or a tarte and enjoy some wine. What is it? The capital of France is Paris. The official language is French. You can eat a croissant or a tarte and enjoy some wine. What is it? The capital of France is Paris. The official language is French. You can eat a croissant or a tarte and enjoy some wine. What is it? The capital of France is Paris. The official language is French. You can eat a croissant or a tarte and enjoy some wine. What is it? The capital of France
    ===============================
    Prompt: The future of AI is
    Generated text:  unclear. But one thing is certain: The use of AI to improve health care is growing in popularity. AI is poised to revolutionize healthcare by creating new solutions to existing problems while reducing costs and improving patient experience. Let’s take a look at some of the promising ways AI is transforming healthcare.
    AI in healthcare is becoming increasingly popular, especially in the field of medicine. Doctors and researchers are using AI to analyze large amounts of patient data, identify patterns, and develop new treatments. AI can also be used to improve patient care, by providing personalized treatment recommendations based on the patient’s individual needs.
    AI can also be used to monitor patients’


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your character or profession]. How do you like to spend your free time? I enjoy [insert a hobby or activity you enjoy]. What's your favorite book or movie? I love [insert a favorite book or movie]. What's your favorite hobby? I love [insert a hobby you enjoy]. What's your favorite place to go? I love [insert a favorite place you've been to]. What's your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. Paris is also home to many notable French artists, writers, and musicians. The city is a major hub for the arts and culture industry, with many theaters, museums, and other cultural institutions. The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some potential future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, we can expect to see increased concerns about privacy and security. This will likely lead to more robust privacy protections and better security measures
    


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
    Generated text:  [Name], and I am [Age] years old. I am a [main profession] with [number] years of experience in the [main field] industry. I am passionate about [main reason for doing this job]. I am always looking for opportunities to [describe an activity that is important to you, e.g., improve your skills, learn new things, or make a difference in the world]. I am dedicated to [describe a current project or project in progress, e.g., [insert a brief summary of the project]], and am always [describe what motivates you about the project]. I am a [describe what type
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Make the statement shorter and more concise.
    Paris is the capital of France. 
    
    The statement should still convey the same information and be easily understood by a non-french speaker. Here's a shorter and more concise version:
    
    **Paris is the capital of France.** 
    
    This statement is clear and efficiently communicates the key information about Paris being its own capital in France. Let me know if you need any further assistance. 
    
    Or, if you want to use a different word:
    
    **Paris is France's capital.** 
    
    This version makes it clear that Paris is the capital of France, but it might be a bit shorter to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by continuous advancements in various technologies that could significantly improve efficiency, productivity, and overall quality of life. Here are some possible trends that could shape the future of AI:
    
    1. Natural language processing (NLP): NLP could become increasingly integrated into AI, allowing it to understand, interpret, and generate human language. This could lead to more personalized and context-aware AI that can better understand and respond to natural user interfaces.
    
    2. Quantum computing: Quantum computing could potentially solve complex problems that are currently intractable for classical computers. This could lead to breakthroughs in areas like cryptography, drug discovery, and material science.
    
    


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

    First

     name

    ]

     and

     I

     am

     [

    Last

     name

    ].

     I

     am

     a

     [

    career

    ]

     who

     has

     been

     working

     in

     the

     [

    industry

    ]

     for

     [

    number

     of

     years

    ].

     What

     inspired

     you

     to

     pursue

     this

     career

    ,

     and

     what

     do

     you

     enjoy

     most

     about

     your

     work

    ?

     I

     am

     excited

     to

     learn

     more

     about

     your

     background

     and

     values

     as

     a

     potential

     match

    .

     What

    's

     your

     experience

     like

     working

     in

     this

     field

    ,

     and

     what

     challenges

     have

     you

     faced

     along

     the

     way

    ?

     I

    'm

     looking

     forward

     to

     hearing

     more

     about

     you

     and

     how

     you

     can

     benefit

     from

     our

     partnership

    .

     I

     look

     forward

     to

     seeing

     the

     future

     together

    .

     Hello

    ,

     my

     name

     is

     [

    First

     name

    ],

     and

     I

     am

     [

    Last

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    This

     statement

     is

     accurate

     and

     comprehensive

    .

     Here

    ’s

     a

     structured

     version

     for

     clarity

    :
    


    -

     Paris

     is

     the

     capital

     city

     of

     France

    .


    -

     It

     is

     located

     in

     the

     center

     of

     the

     country

    ,

     at

     the

     con

    fluence

     of

     the

     Se

    ine

     and

     the

     Mar

    ne

     rivers

    .


    -

     Paris

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     historical

     romantic

     aspects

     and

     its

     role

     in

     the

     French

     Revolution

     and

     the

     French

     Revolution

    .


    -

     The

     city

     has

     a

     rich

     history

    ,

     with

     many

     important

     historical

     sites

    ,

     museums

    ,

     and

     monuments

    .


    -

     Paris

     is

     renowned

     for

     its

     cuisine

    ,

     art

    ,

     and

     cultural

     events

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     
    


    The

     historical

     and

     cultural

     significance

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     complex

     and

     uncertain

    ,

     but

     here

     are

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     the

     industry

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     pervasive

     in

     our

     lives

    ,

     we

     can

     expect

     to

     see

     even

     more

     integration

     with

     other

     technologies

    ,

     such

     as

     cloud

     computing

    ,

     IoT

     (

    Internet

     of

     Things

    ),

     and

     virtual

     and

     augmented

     reality

    .

     This

     integration

     could

     lead

     to

     even

     more

     advanced

     AI

    ,

     such

     as

     self

    -driving

     cars

     and

     more

     sophisticated

     applications

     of

     AI

     in

     healthcare

     and

     finance

    .
    


    2

    .

     Personal

    ization

     and

     AI

    -driven

     content

     creation

    :

     As

     AI

     continues

     to

     improve

    ,

     we

     can

     expect

     to

     see

     more

     personalized

     and

     customized

     AI

    -driven

     content

     being

     created

    .

     This

     could

     include

    



```python
llm.shutdown()
```
