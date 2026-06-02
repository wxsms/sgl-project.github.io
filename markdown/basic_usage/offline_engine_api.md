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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.53it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.48s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.33it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.33it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.72it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.69it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.76it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.74it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.04it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.04it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.04it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.72 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.34it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.34it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.71 GB):  24%|██▍       | 14/58 [00:00<00:01, 28.34it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.68 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.21 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 28.72it/s] Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  38%|███▊      | 22/58 [00:00<00:01, 33.47it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 38.07it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.41it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  55%|█████▌    | 32/58 [00:01<00:00, 41.41it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]

    Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.77it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 45.58it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 46.42it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  90%|████████▉ | 52/58 [00:01<00:00, 47.12it/s] Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  98%|█████████▊| 57/58 [00:01<00:00, 47.72it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 39.46it/s]


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
    Generated text:  Kiyomi and I am a creative artist. I work with a variety of media including painting, drawing, sculpture, ceramics, photography and painting.
    I create artwork as a means of self-expression and as a means of providing a creative outlet for my inner experiences. My work is inspired by my personal experiences and beliefs. I strive to capture the beauty and essence of things, often in unexpected ways. I believe that the more we know about something, the more we realize about ourselves.
    I am also a member of the National Society of Professional Artists (NSPA) and the Art Therapy Association of Canada (ATAC). I strive to use
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is also the leader of the country. He is like the king of the country. The president has a very important job. He is responsible for making important decisions. He also makes sure that all the people in the country know what the president is trying to do. He is like the leader of the country. He may visit other countries. He also makes sure that all the countries in the world know what the president is trying to do. He is like the leader of the country. 
    
    Based on the paragraph above, answer a question. Who makes sure that all the countries in the world know what the president
    ===============================
    Prompt: The capital of France is
    Generated text:  located at the north of the country, it’s also the largest city in France and the capital of a country. The name of the city is Paris. With its history dating back to the 9th century, Paris is the second most populous city in France and one of the 10 largest cities in Europe. Paris is a metropolis with a population of approximately 2.2 million people and it is the cultural and economic center of France, and it is often referred to as the “City of Love”. Paris is famous for the Eiffel Tower, the Louvre, the Notre Dame Cathedral, and the Arc de Tri
    ===============================
    Prompt: The future of AI is
    Generated text:  in space!
    The SpaceX Commercial Crew program has been launching a number of robotic crew capsules into orbit with NASA’s crew transportation vehicle, and we are excited to see what comes next in the future of space exploration. SpaceX is on the verge of another launch, this time with the crew of the Crew Dragon vehicle that will carry astronauts into orbit and onto the International Space Station.
    The Crew Dragon mission, which is scheduled to launch on July 28, 2022, will be the first mission to the International Space Station (ISS) since the crewed space station was launched in 2011.
    The astronauts will be


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and festivals. Paris is a popular tourist destination, known for its rich history, beautiful architecture, and vibrant nightlife. It is also home to the French Parliament, the French National Library, and the French Academy of Sciences. The city is known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and high-tech industries blending with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced decision-making. This could lead to a more human-like experience for users, as AI systems are designed to mimic human behavior and emotions.
    
    2. Enhanced privacy and security: As AI systems become more sophisticated, there is a risk of privacy and security breaches. As a result, there may be increased efforts to improve privacy and security features in AI systems.
    
    3. Increased focus on ethical considerations: As AI systems become more
    


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
    Generated text:  ____. 
    I am a ____. I have been working for ____. For the last ____. I've learned the following skills:
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    - ____. 
    -
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historic significance, cultural richness, and elegant architecture.
    
    The capital of France is Paris, known for its historic significance, cultural richness, and elegant architecture. French cuisine, including French fries and champagne, is a major attraction for tourists, and the city also has a vibrant nightlife scene. Paris has been home to many notable figures throughout history, including writers, artists, and composers. The city is also known for its annual Eiffel Tower celebrations, which attract millions of visitors every year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a combination of increasing complexity, precision, and personalization. Here are some potential trends that are currently underway or are expected to be developed in the coming years:
    
    1. Increased complexity: AI is increasingly complex, with the ability to process large amounts of data and understand complex human behavior. As this complexity increases, it may lead to more sophisticated algorithms that can learn from data and make predictions and recommendations on their own.
    
    2. Precision: AI is becoming more precise in its ability to generate human-like responses. For example, chatbots that understand natural language and can respond to questions in a way that feels human-like are


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

    name

    ],

     and

     I

    'm

     an

     [

    age

    ]

     year

     old

     [

    gender

    ].

     I

    'm

     [

    physical

    特征

    ]

     of

     [

    height

     or

     weight

    ],

     with

     [

    hair

     or

     eyes

    ]

     of

     [

    color

    ].

     I

     have

     [

    occupation

    ]

     and

     love

     [

    occupation

    ],

     with

     [

    number

    ]

     of

     [

    occupation

    ]

     I

    've

     been

     known

     to

     create

    ,

     and

     I

     find

     my

     best

     [

    occupation

    ]

     is

     [

    a

     particular

     one

    ].

     If

     you

    're

     looking

     for

     my

     real

     name

    ,

     I

     don

    't

     have

     one

    .

     However

    ,

     I

    'm

     always

     looking

     to

     make

     new

     friends

     and

     add

     to

     the

     world

     around

     me

    .

     Let

     me

     know

     if

     you

    'd

     like

     to

     know

     more

     about

     me

    !

     [

    name

    ]

     [

    address

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     its

     capital

    .

     It

     is

     located

     in

     the

     Î

    le

    -de

    -F

    rance

     region

    ,

     on

     the

     western

     coast

     of

     the

     French

     Riv

    iera

     and

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

     architecture

    .

     Paris

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Tu

    il

    eries

     Garden

    .

     The

     city

     is

     also

     a

     major

     transportation

     hub

    ,

     with

     a

     complex

     network

     of

     highways

    ,

     railways

    ,

     and

     public

     transit

     systems

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     considered

     one

     of

     the

     most

     liv

    able

     cities

     in

     the

     world

    .

     It

     is

     known

     for

     its

     fashion

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

    ,

     and

     it

    's

     difficult

     to

     predict

     exactly

     what

     technologies

     will

     emerge

     or

     what

     trends

     will

     shape

     the

     field

    .

     However

    ,

     here

     are

     some

     possible

     trends

     that

     could

     potentially

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     Human

    -A

    I

     Collaboration

    :

     As

     AI

     becomes

     more

     advanced

    ,

     we

     are

     likely

     to

     see

     more

     human

    -A

    I

     collaboration

     in

     areas

     like

     healthcare

    ,

     customer

     service

    ,

     and

     law

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     AI

     and

     potentially

     improve

     outcomes

     for

     human

     patients

     and

     businesses

    .
    


    2

    .

     Autonomous

     Systems

    :

     One

     of

     the

     most

     promising

     areas

     of

     AI

     development

     is

     autonomous

     systems

    .

     As

     AI

     continues

     to

     improve

     and

     become

     more

     reliable

    ,

     we

     are

     likely

     to

     see

     more

     autonomous

    



```python
llm.shutdown()
```
