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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.47it/s]


    2026-05-20 12:25:46,162 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-20 12:25:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:03,  4.27s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:17,  2.90it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]Compiling num tokens (num_tokens=960):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s] Compiling num tokens (num_tokens=896):  24%|██▍       | 14/58 [00:04<00:07,  6.27it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:04<00:02, 12.72it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 19.38it/s]

    Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 28.03it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 37.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.57 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.54 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.54 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.26 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.24 GB):   3%|▎         | 2/58 [00:00<00:02, 19.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=73.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.55 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.54 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.54 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.53 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.52 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.52 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.52 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.51 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.50 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.48 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=960 avail_mem=72.50 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s] Capturing num tokens (num_tokens=896 avail_mem=72.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.49 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.00it/s]Capturing num tokens (num_tokens=832 avail_mem=72.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=768 avail_mem=72.49 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=704 avail_mem=72.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=640 avail_mem=72.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=576 avail_mem=72.48 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=512 avail_mem=72.47 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.87it/s]Capturing num tokens (num_tokens=512 avail_mem=72.47 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=480 avail_mem=72.48 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=448 avail_mem=72.48 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=416 avail_mem=72.48 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=384 avail_mem=72.48 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=320 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:00<00:00, 43.10it/s]Capturing num tokens (num_tokens=320 avail_mem=72.04 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=288 avail_mem=72.04 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  60%|██████    | 35/58 [00:00<00:00, 45.74it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  60%|██████    | 35/58 [00:01<00:00, 45.74it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]

    Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=112 avail_mem=72.01 GB):  71%|███████   | 41/58 [00:01<00:00, 47.26it/s]Capturing num tokens (num_tokens=112 avail_mem=72.01 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s] Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  79%|███████▉  | 46/58 [00:01<00:00, 47.86it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  88%|████████▊ | 51/58 [00:01<00:00, 46.76it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.69it/s] Capturing num tokens (num_tokens=4 avail_mem=71.97 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.69it/s]Capturing num tokens (num_tokens=4 avail_mem=71.97 GB): 100%|██████████| 58/58 [00:01<00:00, 41.33it/s]


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
    Generated text:  Ruben. My favorite color is brown. That’s what I like to wear. I like to sing and I also like to dance. I like to listen to music. I also like to be with my family and friends. I like to go to the park on Saturdays. That is a nice place to have fun. The best part about the park is that you can see the world. I am very tall and I also like to play sports. I like to play soccer. My favorite sport is basketball because it is fun to play. I also like to eat pizza. I like to eat pizza with my family. My favorite food
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to estimate the average height of all adults in the country. To gather this information, the president decides to randomly select a sample of 50 adults from the population. The president assumes that the average height of all adults in the country is 170 cm with a standard deviation of 10 cm. Assuming the heights are normally distributed, what is the probability that the sample mean height will be greater than 173 cm?
    To determine the probability that the sample mean height will be greater than 173 cm, we can use the properties of the normal distribution. Here are the steps:
    
    1. **Ident
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. 
    A. Paris
    B. Marseille
    C. Toulouse
    D. Nice
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. Marseille
    C. Toulouse
    D. Nice
    Answer:
    
    A
    
    The capital of France is ____
    A. Paris
    B. Marseille
    C. Toulouse
    D. Nice
    Answer:
    
    A
    
    In the book "The History of the Egyptians, Indians, and Assyrians," it is recorded: "There once was a ruler in India who was very wise. He made a promise to give his most favored subjects the best of everything for
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the potential for unintended consequences remains. In the short term, we can expect a surge in the use of AI across all industries. In the long term, AI will continue to evolve, and we can expect to see improvements in the areas of facial recognition, autonomous vehicles, and predictive analytics. But the potential for unintended consequences remains. It's important for organizations to carefully consider the potential risks and develop strategies to mitigate them.
    One of the major challenges in the development of AI is the ethical and social implications of its use. There is a risk that AI may be used to perpetuate or exacerbate existing social inequalities, or that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for being] and I'm always looking for ways to [benefit to the company]. I'm a [reason for being] and I'm always looking for ways to [benefit to the company]. I'm a [reason for being] and I'm always looking for ways to [benefit to the company]. I'm a [reason for being] and I'm always looking for ways to [benefit to the company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is a bustling city with a rich history and culture, and it is a popular tourist destination. The city is also known for its fashion industry, with many famous fashion houses and boutiques located in the city. Overall, Paris is a city of contrasts and excitement, and it is a must-visit destination for anyone interested in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and artificial general intelligence: As AI continues to improve, it is likely to become more capable of performing a wide range of tasks, including decision-making, problem-solving, and creative thinking. This could lead to the development of AI that can perform tasks that are currently done by humans, such as driving, writing, and playing sports.
    
    2. Increased reliance on AI for decision-making: As AI becomes more capable of making decisions, it is likely to become more prevalent
    


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
    Generated text:  [Name], and I'm a professional who specializes in [specialization]. I am a [position], and my [professional title] has been with me since [year], and I love [my job or profession]. I am always eager to learn new things and keep up with the latest trends and best practices in my field. I am also an [personality trait], and I am always ready to help others grow and learn. I am passionate about [my personal interests or hobbies], and I love spending time with my family, friends, and animals. I am also someone who is [a trait or quality], and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its rich history, art, and stunning architecture. The city is also famous for its vibrant nightlife and French cuisine. Paris is a popular destination for tourists and locals alike, and its culture and history are celebrated worldwide. The city is located in the center of the country and is known for its famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Montmartre. Paris is the capital of France and is home to many of the country's cultural and historical landmarks. With its unique blend of art, culture, and history, Paris is a city that is steeped in history and tradition
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and potential. Here are some potential trends that are currently being explored:
    
    1. Autonomous vehicles: With advancements in AI, autonomous vehicles will become more common, potentially reducing traffic accidents and improving traffic flow on highways.
    
    2. Personalized medicine: AI will enable more accurate diagnoses and treatment plans, leading to better health outcomes for patients.
    
    3. Voice assistants: AI will continue to evolve, with more advanced models becoming available to consumers. These assistants will be able to understand speech, text, and images, making them more personal and convenient for users.
    
    4. Financial fraud detection: AI will be used to detect fraudulent activities,


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

     your

     name

    ]

     and

     I

    'm

     a

     [

    insert

     profession

     or

     hobby

    ].

     I

    'm

     excited

     to

     have

     the

     opportunity

     to

     share

     my

     experiences

     and

     knowledge

     with

     you

     all

    .

     I

    'm

     here

     to

     help

     you

     with

     whatever

     you

     need

    .

     Let

    's

     get

     started

    !

     Let

    's

     talk

     about

     our

     day

    !

     Welcome

     to

     [

    insert

     your

     name

    ].

     This

     is

     [

    insert

     a

     neutral

     expression

    ]

     about

     you

    .

     What

     brings

     you

     here

     today

    ?

     [

    insert

     your

     name

    ].

     I

     hope

     you

    're

     here

     to

     learn

     something

     new

     and

     share

     your

     knowledge

     with

     me

    .

     What

     do

     you

     hope

     to

     gain

     from

     this

     session

    ?

     [

    insert

     your

     name

    ].

     You

    've

     been

     an

     avid

     fan

     of

     [

    insert

     a

     hobby

     or

     interest

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Okay

    ,

     remember

     the

     big

     city

     in

     the

     south

     of

     France

    ?

     That

    's

     Paris

    !

     It

    's

     the

     most

     famous

     and

     famous

     city

     in

     France

    .

     Paris

     is

     very

     big

     and

     has

     lots

     of

     fancy

     buildings

     and

     pretty

     streets

    .

     Some

     of

     the

     best

     things

     to

     see

     there

     are

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

     Paris

     is

     a

     very

     interesting

     place

     to

     visit

    !

     Have

     you

     ever

     been

     there

    ?

     Let

     me

     know

     if

     you

     want

     to

     know

     about

     Paris

     in

     more

     detail

    .

     I

    'll

     do

     my

     best

     to

     answer

     as

     faithfully

     as

     I

     can

    !

     What

     do

     you

     think

     of

     Paris

    ?

     Is

     it

     fun

     or

     boring

    ?

     The

     best

     way

     to

     answer

     your

     question

     is

     to

     tell

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     unpredictable

     and

     complex

    ,

     but

     some

     possible

     trends

     that

     are

     currently

     being

     studied

     and

     explored

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     is

     being

     used

     more

     and

     more

     in

     the

     healthcare

     industry

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     outcomes

    .

     For

     example

    ,

     AI

     is

     being

     used

     to

     analyze

     medical

     images

     such

     as

     X

    -rays

     and

     MR

    Is

     to

     detect

     cancer

    ,

     track

     patient

     progression

    ,

     and

     predict

     treatment

     responses

    .
    


    2

    .

     Enhanced

     AI

     ethics

     and

     accountability

    :

     As

     AI

     systems

     become

     more

     integrated

     into

     everyday

     life

    ,

     there

     is

     growing

     concern

     about

     the

     ethical

     implications

     of

     AI

     systems

    .

     For

     example

    ,

     there

     are

     concerns

     about

     bias

     in

     AI

     algorithms

    ,

     the

     potential

     for

     AI

    



```python
llm.shutdown()
```
