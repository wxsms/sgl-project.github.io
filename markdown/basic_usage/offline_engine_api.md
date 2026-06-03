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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:54,  5.16s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:54,  5.16s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:54,  5.16s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:54,  5.16s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:54,  5.16s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:42,  1.26it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.78it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]

    Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.04it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 12.72it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]

    Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 18.47it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 25.92it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 35.38it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 35.38it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 35.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.73 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):   9%|▊         | 5/58 [00:00<00:02, 20.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.57it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s]Capturing num tokens (num_tokens=960 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s] Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s]Capturing num tokens (num_tokens=768 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.23it/s]Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=416 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:00<00:00, 36.19it/s]Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.19it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.19it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  60%|██████    | 35/58 [00:01<00:00, 37.04it/s]Capturing num tokens (num_tokens=208 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=192 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  69%|██████▉   | 40/58 [00:01<00:00, 38.66it/s]Capturing num tokens (num_tokens=128 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s] Capturing num tokens (num_tokens=80 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.36it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=32 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.61it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=8 avail_mem=74.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.13it/s] Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.13it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 36.99it/s]


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
    Generated text:  Sophie and I am an experienced chatbot with expertise in creating virtual assistants that are well-suited for varied tasks. I am skilled at crafting personalized content, engaging with users in a meaningful way, and leveraging natural language processing to understand and respond to users' queries. I am a member of a team that is committed to continuously improving the quality and effectiveness of my services. I am always looking for ways to enhance my capabilities and to improve the user experience with my virtual assistants. So if you have any questions or need assistance with any of my services, feel free to ask! 🌟✨✨✨
    
    What kind of tasks can a
    ===============================
    Prompt: The president of the United States is
    Generated text:  currently 37 years old. 7 years ago, he was 23 years old. What would be his age 15 years from now?
    To determine the president's current age, we start by using the information given about his age 7 years ago. We know that he was 23 years old at that time. Therefore, we can set up the following equation:
    
    \[
    \text{Current age} - 7 = 23
    \]
    
    Adding 7 to both sides of the equation, we get:
    
    \[
    \text{Current age} = 23 + 7 = 3
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. There are many famous landmarks in Paris, including the Eiffel Tower, the Louvre, Notre Dame Cathedral, and the Notre Dame Cathedral. The Louvre is also a museum that houses over 250,000 works of art from all over the world. In Paris, you can enjoy a variety of food and drinks, from traditional French dishes like croissants and baguettes, to modern French cuisine like steak and oysters. The city also has a rich history and culture, with a vibrant arts scene and a reputation for being a popular tourist destination. No matter what type of activity you're planning
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but it's not just about the technology. AI will have a significant impact on society, and it will be influenced by many factors.
    
    AI will have a huge impact on business, with more jobs disappearing and others being created. It will also have a huge impact on privacy and security, with new risks emerging.
    
    AI will also have a significant impact on education, with online learning becoming the norm. It will also have a significant impact on healthcare, with AI being used to diagnose and treat diseases.
    
    AI will also have a significant impact on the environment, with more automation and environmental impact reducing.
    
    Overall, the future of AI is bright


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and experiences. Let's chat! [Name] [Company Name] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Instagram Profile] [GitHub Profile] [LinkedIn Profile] [Twitter
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French Parliament House. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the French Parliament, the French Parliament House, and the Eiffel Tower. Paris is a vibrant and dynamic city that is a must-visit destination for anyone interested in French culture and history. The city is also known for its fashion and art scene
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency and productivity, but it could also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, we will need to address the ethical and privacy concerns that come with it. This could lead to new regulations and standards being developed to ensure
    


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
    Generated text:  [Your Name], and I'm a [job title] at [Company Name]. I've been with the company for [number of years] years. Here are a few things you can expect from me: [fill in the blank]. (Note: You can choose to fill in the blank with a specific detail that represents your interests or skills in addition to your job title and company name.) I'm looking forward to working with you and I'm excited to help you succeed. What can you tell me about yourself? [Tell me about your personality, strengths, weaknesses, and hobbies]. I look forward to hearing about you and how you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and is located in the south of the country. It was founded in the 8th century and has been the center of French culture, politics, and industry for centuries. It is famous for its romantic architecture, vibrant nightlife, and annual couture fashion show. Paris is a major cultural and economic hub and is home to the Eiffel Tower, Louvre Museum, and many other famous landmarks. Despite its historical importance, Paris is now a popular tourist destination and a symbol of French culture and identity. 
    
    1. Paris is located in the south of France.
    2. It was
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by rapid and transformative changes, driven by advances in computing power, data analysis, and machine learning. Here are some potential trends in AI that are likely to shape the field in the next few years:
    
    1. Increased focus on ethical AI: As AI becomes more integrated into society, there is a growing recognition of the need to address potential ethical concerns. This may lead to increased focus on AI's impact on society and the development of new ethical guidelines and standards.
    
    2. Increased use of AI for healthcare: AI is increasingly being used in healthcare to diagnose and treat diseases, predict patient outcomes, and identify potential medical risks.


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

    Occup

    ation

    ]

     who

     is

     passionate

     about

     [

    Career

     Goal

    ]

     and

     always

     [

    Mot

    ivation

    /

    Ad

    vent

    urous

     Att

    itude

    ].

     I

     enjoy

     [

    A

     hobby

     or

     sport

     of

     yours

    ]

     and

     I

     am

     constantly

     learning

     new

     things

     and

     trying

     new

     things

    .

     I

     am

     always

     eager

     to

     make

     a

     positive

     impact

     in

     the

     world

     and

     to

     help

     others

    .

     I

     am

     always

     motivated

     by

     a

     desire

     to

     explore

     new

     opportunities

     and

     challenges

    .

     I

     believe

     that

     with

     hard

     work

     and

     persistence

    ,

     anything

     is

     possible

    .

     How

     would

     you

     describe

     your

     style

     of

     communication

    ?

     I

     am

     someone

     who

     communicates

     in

     a

     concise

     and

     clear

     manner

    ,

     using

     simple

     language

     to

     explain

     complex

     ideas

     and

     concepts

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     and

     most

     populous

     city

     in

     the

     country

     and

     is

     located

     in

     the

     south

     of

     France

    .


    Paris

    ,

     the

     largest

     and

     most

     populous

     city

     in

     France

    ,

     is

     located

     in

     the

     southern

     part

     of

     the

     country

     and

     is

     known

     for

     its

     historic

     landmarks

    ,

     vibrant

     culture

    ,

     and

     annual

     festivals

    .

     The

     city

     is

     also

     home

     to

     several

     major

     universities

    ,

     including

     the

     University

     of

     Paris

    ,

     and

     is

     a

     major

     center

     for

     international

     trade

    ,

     finance

    ,

     and

     business

    .

     As

     of

     

    2

    0

    2

    1

    ,

     the

     population

     of

     Paris

     was

     estimated

     at

     around

     

    2

    .

    2

     million

     people

    ,

     making

     it

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .

     Paris

     is

     a

     cultural

     and

     economic

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     with

     a

     variety

     of

     trends

     shaping

     the

     technology

     we

     use

     every

     day

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Rob

    ust

    ness

     and

     Security

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     they

     will

     require

     more

     robust

     security

     measures

     to

     protect

     against

     threats

     and

     attacks

    .

     This

     will

     involve

     developing

     new

     algorithms

    ,

     techniques

    ,

     and

     systems

     to

     detect

     and

     prevent

     malicious

     activities

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     become

     even

     more

     personalized

    ,

     allowing

     users

     to

     receive

     recommendations

     based

     on

     their

     behavior

     and

     preferences

    .

     This

     will

     enable

     more

     efficient

     and

     effective

     use

     of

     resources

     in

     the

     data

     center

     and

     will

     allow

     for

     more

     targeted

     marketing

    .
    


    3

    .

     Artificial

     Intelligence

     in

     Health

     Care

    :

     AI

    



```python
llm.shutdown()
```
