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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:54,  4.11s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.57it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.68it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.88it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 16.18it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.51it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:04<00:00, 31.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:03, 18.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:03, 18.46it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 21.48it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  14%|█▍        | 8/58 [00:00<00:03, 16.58it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.67it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.67it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  17%|█▋        | 10/58 [00:00<00:02, 16.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.21it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.21it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.21it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.21it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.38it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.38it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.16it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.16it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.16it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.16it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 31.16it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  50%|█████     | 29/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.42it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.70it/s]

    Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 33.70it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=144 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  71%|███████   | 41/58 [00:01<00:00, 28.17it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=96 avail_mem=74.28 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.40it/s] Capturing num tokens (num_tokens=80 avail_mem=73.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.40it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=48 avail_mem=74.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 32.40it/s]Capturing num tokens (num_tokens=48 avail_mem=74.24 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=32 avail_mem=74.02 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=28 avail_mem=74.01 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=24 avail_mem=74.23 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.03it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.22 GB):  86%|████████▌ | 50/58 [00:01<00:00, 29.03it/s]Capturing num tokens (num_tokens=20 avail_mem=74.22 GB):  93%|█████████▎| 54/58 [00:01<00:00, 27.62it/s]Capturing num tokens (num_tokens=16 avail_mem=74.21 GB):  93%|█████████▎| 54/58 [00:01<00:00, 27.62it/s]Capturing num tokens (num_tokens=12 avail_mem=74.20 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.62it/s]Capturing num tokens (num_tokens=8 avail_mem=74.19 GB):  93%|█████████▎| 54/58 [00:02<00:00, 27.62it/s] Capturing num tokens (num_tokens=8 avail_mem=74.19 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.37it/s]Capturing num tokens (num_tokens=4 avail_mem=74.19 GB):  98%|█████████▊| 57/58 [00:02<00:00, 27.37it/s]Capturing num tokens (num_tokens=4 avail_mem=74.19 GB): 100%|██████████| 58/58 [00:02<00:00, 27.05it/s]


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
    Generated text:  Daniela and I am from Huelva in Spain. I love to eat and drink, especially Spanish beer. I also love to travel, learn new things and make new friends. I was born in 2004 in Mexico and I am currently studying at UC Berkeley to become a PhD student. When I am not working, you can find me playing basketball or running marathons. I have a passion for humor and am a comedy writer. When I'm not writing comedy, you can find me exploring the outdoors, surfing or practicing yoga. I'm a multilingual and I speak English, Spanish, French and Spanish.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a major figure in the country and will be seen as a “world leader” and an important national symbol. As such, the current president has received the most media attention of all the U.S. presidents in his lifetime. However, the office of president of the United States has been relatively minor and not as glamorous as many other offices. This was true even in the days of the monarchy, as the British monarchs did not have the same level of power in the United States.
    
    Based on the above statement, what can be concluded about the U.S. President?
    
    OPTIONS: (A). He has not been received with much media attention
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Lyon
    C. Nice
    D. Marseille
    Answer:
    A
    
    Which of the following statements about the 'water tower effect' and the 'resource curse' is correct?
    A. Both are specific cases of development failure.
    B. Both are specific cases of economic decline.
    C. The 'water tower effect' occurs in developed countries, while the 'resource curse' occurs in developing countries.
    D. The 'water tower effect' occurs in developing countries, while the 'resource curse' occurs in developed countries.
    Answer:
    D
    
    Which of the following statements about the 'water tower effect' and
    ===============================
    Prompt: The future of AI is
    Generated text:  taking us somewhere very exciting. The way we interact with the world is changing, and one of the ways it's doing that is through the influence of AI. As AI has been making significant progress, it's becoming increasingly common to hear about the potential for AI in virtually every industry, from healthcare, finance, and education to entertainment, and beyond. The potential benefits of AI are vast, and it's clear that it has the potential to transform the way we work and live in the future.
    One of the biggest challenges facing AI is how to make sure that it is being used ethically and responsibly. This is particularly important given the potential


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a cultural and economic center with a rich history and a diverse population. The city is home to many famous French artists, writers, and musicians, and is a major hub for business, politics, and entertainment. Paris is a popular tourist destination and a major center for international business and diplomacy. The city is also known for its cuisine, with dishes such as croissants,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools could be used to identify diseases earlier and more accurately, and AI-powered treatments could be used to personalize treatment plans for patients.
    
    2. Increased use of AI in finance: AI is already being used in finance to help manage risk and make
    


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
    Generated text:  [Your Name] and I'm a friendly and helpful assistant. I'm always here to assist anyone who needs a little extra help with their needs. My name is [Your Name] and I'm a friendly and helpful assistant. I'm always here to assist anyone who needs a little extra help with their needs. You can trust me to be there to help you with whatever you need. Just let me know what you need and I'll do my best to assist you. If you're looking for something specific, please let me know and I'll do my best to find it for you. Whether you need help with something trivial or something
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and the largest city in the world by population. It is also the seat of the government, the capital of the government, the legislative and executive capital of France. Paris is known for its iconic Eiffel Tower, Notre Dame Cathedral, the Louvre Museum, the Arc de Triomphe, the Champs-Élysées, and the Louvre. It is a hub for art, culture, and commerce. Its architecture is renowned for its romantic and ornate style. Paris is also one of the world's most popular tourist destinations. The city is known for its rich history and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to evolve rapidly, with many potential developments that could dramatically change the way we live, work, and interact with technology. Here are some of the key trends that could influence AI in the coming years:
    
    1. Advancements in machine learning: With advances in machine learning algorithms, AI systems will be able to learn from and adapt to new data more efficiently. This could lead to more complex and sophisticated AI systems that can perform a wider range of tasks, from image and speech recognition to autonomous vehicles and personalized medicine.
    
    2. Increased focus on ethical AI: As more and more AI systems become available, there will be a greater emphasis


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

     [

    Age

    ].

     I

     am

     a

     [

    occupation

    ]

     who

     has

     lived

     in

     [

    country

    ]

     for

     [

    number

     of

     years

    ].

     I

     am

     [

    career

     motivation

    ]

     driven

    .

     I

     have

     a

     passion

     for

     [

    what

     you

     like

     to

     do

    ].

     I

     am

     confident

     in

     my

     [

    strength

    s

     or

     skills

    ].

     What

     exc

    ites

     you

     most

     about

     my

     character

    ?

     [

    What

     exc

    ites

     you

     most

     about

     me

    ]?

     


    [

    Your

     answer

    ]

     


    Tell

     me

     more

     about

     yourself

    ,

     [

    Name

    ],

     and

     how

     you

     became

     who

     you

     are

    .

     [

    Your

     answer

    ]


    What

     is

     your

     favorite

     hobby

     or

     activity

    ?

     [

    Your

     answer

    ]


    What

     do

     you

     enjoy

     doing

     in

     your

     free

     time

    ?

     [

    Your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    "

     and

     is

     home

     to

     iconic

     landmarks

     such

     as

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
    


    The

     city

    's

     vibrant

     culture

    ,

     rich

     history

    ,

     and

     beautiful

     architecture

     make

     it

     an

     attractive

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     is

     also

     known

     for

     its

     culinary

     scene

    ,

     including

     famous

     dishes

     like

     cro

    iss

    ants

     and

     esc

    arg

    ot

    .

     The

     city

     is

     a

     major

     hub

     for

     the

     arts

    ,

     with

     many

     museums

    ,

     galleries

    ,

     and

     theaters

    ,

     and

     is

     a

     major

     transportation

     hub

     for

     Europe

    .

     Overall

    ,

     Paris

     is

     a

     unique

     and

     fascinating

     city

     that

     offers

     a

     wealth

     of

     experiences

     and

     attractions

     to

     explore

    .

     
    


    There

     is

     no

     shortage

     of

     activities

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     incredibly

     exciting

     and

     dynamic

    ,

     with

     a

     wide

     range

     of

     potential

     trends

     that

     could

     shape

     how

     we

     use

     and

     develop

     AI

     technology

     in

     the

     years

     to

     come

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     could

     be

     expected

     in

     the

     AI

     field

     in

     the

     coming

     years

    :
    


    1

    .

     The

     rise

     of

     edge

     AI

    :

     Edge

     AI

     refers

     to

     the

     use

     of

     AI

     technology

     on

     devices

     that

     are

     close

     to

     the

     user

    ,

     such

     as

     mobile

     devices

    ,

     laptops

    ,

     and

     smartphones

    .

     This

     type

     of

     AI

     has

     the

     potential

     to

     deliver

     faster

     and

     more

     efficient

     processing

     power

     than

     traditional

     cloud

    -based

     AI

    ,

     making

     it

     possible

     to

     create

     systems

     that

     can

     handle

     high

     volumes

     of

     data

     and

     operate

     in

     real

    -time

    .
    


    2

    .

     AI

     in

    



```python
llm.shutdown()
```
