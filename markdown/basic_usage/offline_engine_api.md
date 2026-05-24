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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.75s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:50,  1.08it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:19,  2.63it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2560):  14%|█▍        | 8/58 [00:05<00:19,  2.63it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s] 

    Compiling num tokens (num_tokens=896):  26%|██▌       | 15/58 [00:05<00:06,  6.22it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 11.39it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]

    Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 16.74it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 23.99it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]

    Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 31.88it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 41.12it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 41.12it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 41.12it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 41.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.21 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.88it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.20it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.14 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.96it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.13 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  29%|██▉       | 17/58 [00:00<00:01, 29.96it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=960 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=832 avail_mem=73.63 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=768 avail_mem=73.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=704 avail_mem=73.62 GB):  36%|███▌      | 21/58 [00:00<00:01, 30.26it/s]Capturing num tokens (num_tokens=704 avail_mem=73.62 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.89it/s]Capturing num tokens (num_tokens=640 avail_mem=73.61 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.89it/s]

    Capturing num tokens (num_tokens=576 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.89it/s]Capturing num tokens (num_tokens=512 avail_mem=73.44 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.89it/s]Capturing num tokens (num_tokens=480 avail_mem=73.46 GB):  45%|████▍     | 26/58 [00:00<00:00, 33.89it/s]Capturing num tokens (num_tokens=480 avail_mem=73.46 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.03it/s]Capturing num tokens (num_tokens=448 avail_mem=73.46 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.03it/s]Capturing num tokens (num_tokens=416 avail_mem=73.45 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.03it/s]Capturing num tokens (num_tokens=384 avail_mem=73.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=352 avail_mem=73.45 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=320 avail_mem=73.44 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.03it/s]Capturing num tokens (num_tokens=320 avail_mem=73.44 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=288 avail_mem=73.44 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]

    Capturing num tokens (num_tokens=256 avail_mem=73.44 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=240 avail_mem=73.43 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=224 avail_mem=73.43 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=208 avail_mem=73.42 GB):  60%|██████    | 35/58 [00:01<00:00, 37.68it/s]Capturing num tokens (num_tokens=208 avail_mem=73.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=192 avail_mem=73.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=176 avail_mem=73.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=160 avail_mem=73.42 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=144 avail_mem=73.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.71it/s]

    Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=112 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=96 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s] Capturing num tokens (num_tokens=80 avail_mem=73.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.83it/s]Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=32 avail_mem=73.39 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]

    Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.38it/s]Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.06it/s] Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 36.22it/s]


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
    Generated text:  Christina and I am a Marketing and Public Relations student at the University of Pennsylvania. I am currently a peer mediator. My role in the city is to bring together the city and the community and ensure that our team is working together in a positive way. I would like to ask you all to take a moment to think about how we can promote the message that we work with to the public in a positive way. I would like to know the role that each team member plays in the project. Can you please list the roles of each team member? The role of the marketing team is to ___________ the message. The role of the public
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. The president has many important jobs and tasks, such as being a leader of the country, making important laws, and running the country. The president can change the laws of the country, so that everyone in the country has the same rules and laws, or they can choose to change laws that affect some people, such as the way things are sold, and make changes to other people's lives. Sometimes the president can change laws, even when other people disagree. The president can also make important decisions for the country, such as running the country and making important decisions, even when other people disagree. As the president is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in the European Union and the largest city in Europe by population. There are many famous landmarks and cultural attractions in Paris, such as the Louvre Museum, the Eiffel Tower, the Notre-Dame Cathedral, and the Arc de Triomphe. The city is also home to many art museums, including the Musée d'Orsay, the Musée d'Orsay, the Musée d'Orsay, and the Musée d'Orsay. Paris is a popular tourist destination, and many tourists visit the city for its beautiful scenery, delicious cuisine, and rich cultural heritage.
    ===============================
    Prompt: The future of AI is
    Generated text:  here… but it’s not going to be that way
    
    This article was originally published on Bloomberg BusinessWeek.
    
    In a few months, it’s going to be the year the smartphones, watches, and other digital devices that we use every day become 100 percent artificial intelligence.
    
    In an article published in the latest issue of “The New York Times,” the New York Times’ editorial board suggests that by 2030, artificial intelligence will account for 10 percent of the US economy.
    
    That’s an astounding achievement, and a significant one. But, in reality, that’s going to be a mistake. There’s


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the industry]. I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [action or goal]. I'm a [reason for interest in the industry] and I'm always looking for ways to [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 10 million people. The city is known for its fashion industry, art scene, and food culture, and is a popular tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also home to many international organizations and institutions, including the European Union and the United Nations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Integration of AI with other technologies: AI is already being integrated into a wide range of technologies, including healthcare, finance, and transportation. As AI becomes more integrated into other technologies, we can expect to see even more integration.
    
    3
    


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
    Generated text:  [Your Name] and I’m a [Your Profession] with a keen interest in [Your Area of Expertise or Passion]. I thrive on creating content that helps others achieve their goals and grow in their career. Whether it's helping people find the perfect job or teaching them how to develop their skills, I aim to make the world a better place through my words. In my free time, I enjoy writing short stories and editing books. What’s your favorite hobby, and what are some books or TV shows you like to watch? As an AI language model, I don't have personal hobbies or interests, but I have learned many things
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe by population and is known for its rich history, art, cuisine, and fashion. French cuisine, known for its famous dishes like croissants and crêpes, is a popular tourist attraction. Paris is also home to the Eiffel Tower, one of the most recognizable landmarks in the world. It is home to many of the world’s leading museums, including the Louvre, the Metropolitan Museum of Art, and the Tate Modern. Paris is a bustling city with a diverse and unique culture, and it continues to grow and evolve as a major global city.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by significant advancements and innovations in several areas. Here are some possible trends that are likely to shape the AI landscape in the coming years:
    
    1. Increased Integration of AI with Other Technologies: AI is already making significant contributions to various other fields, such as healthcare, finance, and security. In the future, we may see AI integrated with other technologies, such as blockchain, quantum computing, and robotics, to develop new and innovative solutions.
    
    2. Greater Use of Explainable AI: As AI becomes more integrated into our lives, we may see greater emphasis on developing AI that is more transparent and explainable. This will require


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

    ],

     and

     I

     am

     a

     [

    insert

     occupation

    ]

     with

     [

    insert

     number

     of

     years

     since

     graduating

    ].

     Throughout

     my

     academic

     career

    ,

     I

     have

     always

     been

     passionate

     about

     [

    insert

     something

     you

     enjoy

     doing

     or

     thinking

     about

    ,

     such

     as

     playing

     a

     sport

    ,

     reading

    ,

     or

     cooking

    ].

     I

     am

     constantly

     motivated

     by

     a

     desire

     to

     learn

    ,

     challenge

     myself

    ,

     and

     make

     a

     positive

     impact

    .

     I

     enjoy

     working

     with

     [

    insert

     a

     colleague

     or

     team

     member

    ],

     as

     they

     have

     provided

     me

     with

     valuable

     feedback

     and

     support

    .

     I

     am

     excited

     to

     learn

     more

     about

     you

     and

     what

     you

     bring

     to

     the

     table

    .

     How

     can

     I

     best

     meet

     you

    ?

     [

    insert

     how

     you

     can

     meet

     the

     other

     person

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     cultural

     and

     historical

     heritage

    .

     
    


    (

    Explanation

    :

     This

     response

     conc

    is

    ely

     communicates

     the

     key

     facts

     about

     Paris

    ,

     including

     its

     status

     as

     the

     capital

    ,

     historical

     significance

    ,

     and

     cultural

     heritage

    .)

     
    


    -

     It

     is

     the

     capital

     of

     France

    .


    -

     It

     is

     located

     in

     the

     heart

     of

     France

     and

     has

     a

     population

     of

     approximately

     

    2

    .

    1

     million

     people

    .


    -

     The

     city

     is

     home

     to

     many

     notable

     landmarks

     and

     attractions

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

     various

     museums

     and

     historic

     sites

    .

     
    


    (

    Explanation

    :

     This

     response

     provides

     additional

     information

     about

     the

     attractions

     in

     Paris

     and

     the

     importance

     of

     each

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     diverse

     and

     exciting

    ,

     with

     potential

     areas

     of

     development

     that

     include

    :
    


    1

    .

     Enhanced

     AI

     capabilities

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     more

     intelligent

     and

     adaptable

     systems

     that

     can

     handle

     a

     wide

     range

     of

     tasks

     and

     adapt

     to

     changing

     environments

    .
    


    2

    .

     Self

    -aware

     AI

    :

     In

     the

     coming

     years

    ,

     we

     may

     see

     the

     development

     of

     AI

     systems

     that

     exhibit

     consciousness

     and

     self

    -aware

    ness

    ,

     capable

     of

     understanding

     and

     learning

     from

     their

     environment

     and

     making

     decisions

     that

     are

     self

    -contained

     and

     self

    -exec

    uting

    .
    


    3

    .

     Autonomous

     systems

    :

     Autonomous

     AI

     systems

     are

     systems

     that

     are

     designed

     to

     operate

     without

     human

     intervention

    ,

     which

     could

     revolution

    ize

     many

     industries

     such

     as

     transportation

    ,

     agriculture

    ,

     and

    



```python
llm.shutdown()
```
