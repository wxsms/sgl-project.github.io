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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:41,  4.94s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.95it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.39it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 13.28it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 19.18it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.03it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 33.95it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 33.95it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 33.95it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 33.95it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 33.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:04, 13.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:04, 13.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:03, 16.48it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:03, 16.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 16.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:03, 16.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.50it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.50it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 25.94it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 25.94it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.44it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.80it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.80it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.80it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  34%|███▍      | 20/58 [00:00<00:01, 31.80it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 33.15it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.15it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 33.15it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.89it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 34.45it/s]

    Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 34.45it/s]Capturing num tokens (num_tokens=208 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.32it/s]Capturing num tokens (num_tokens=192 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.32it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.32it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.32it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 34.32it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.81it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.81it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.81it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.86it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.86it/s]Capturing num tokens (num_tokens=48 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.86it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.86it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.86it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s]

    Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 34.11it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 36.31it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 31.83it/s]


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
    Generated text:  Sarah, and I'm a computer scientist. As a leading AI language model, I can analyze and interpret text data to understand patterns, trends, and relationships between different pieces of information. With the help of my programming, I can assist users in generating text, answering questions, and performing various tasks related to text analysis. How can I assist you today? Feel free to ask me any question or task related to text analysis or any other topic! Remember to share any specific details or context that can help me better understand your needs. Let's get started! 🤖💻💻 #textanalytics #AI #computerscience #technology
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to strengthen the country’s military by preparing for a potential war against Iran. Obama announced in 2009 that the United States would deploy a new ship called USS John C. Stennis to the Persian Gulf.
    President Obama announced that the USS John C. Stennis would be the first ship to be deployed to the Persian Gulf to take part in a covert military operation.
    The ship is part of the US Navy’s “Joint Special Operations Command” (JSOC).
    The ship is being used in a covert operation to locate and destroy any of the Iranian weapons of mass destruction in the region. The operation was called Operation Iraqi
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is a city famous for its architecture, its museums, and the famous Eiffel Tower. The city is located in the western part of France and on the north bank of the Seine river. The capital of France is also the capital of the region of Île-de-France, and is known as the "Grand-Landsberg" as it is on a high ground.
    
    The city of Paris is not only famous for its architecture, its museums, and the famous Eiffel Tower, but it is also known for its historical landmarks, including the Notre-Dame Cathedral, the Louvre Museum, the Faub
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it is rapidly transforming our way of life. We are seeing a number of new applications of AI that are not only enabling new technologies like robotics and autonomous vehicles, but also providing us with a way to improve our lives in new ways.
    One of the most exciting areas of AI is the field of healthcare. AI has the potential to revolutionize the way we think about and treat diseases, and it has already shown promise in fields like cancer, neurology, and genetic testing.
    In recent years, AI has been used in a variety of ways in healthcare, including in the development of better diagnostic tools, in the prediction and prevention


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other attractions. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French culture. Paris is also home to many famous French artists, writers, and musicians. The city is a major center for the arts, with many museums, theaters, and other cultural institutions.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy. AI developers will need to be more mindful of the potential consequences of their work and work to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence in the future.
    


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
    Generated text:  [Name]. I'm a [Level] professional with [Skill Set or Specialty] and [Location] experience. I'm known for my [Strengths or Unique Selling Points] and [Areas of Strength]. I'm always ready to learn and grow, and am always eager to help others achieve their goals. Whether you're looking for guidance on [Purpose or Goal], or simply looking to connect with a friendly face, I'm here to assist. Please let me know what you'd like to talk to me about. How can I assist you today? What do you want to know or talk about? Let's get started. [Name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the most populous city and the capital of France. Paris is known for its architecture, gastronomy, and unique culture. The city is home to iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also famous for its annual Paris Festival, which brings together music, dance, and theater from around the world. Paris has a rich cultural scene with many museums, theaters, and festivals throughout the year. It is one of the most visited cities in the world, with millions of visitors each year. Paris is also a global center of finance, business
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a wide range of factors, including advances in computing power, data analysis, and machine learning techniques. Here are some possible trends that are currently being explored:
    
    1. Increased sophistication: AI is currently capable of performing a wide range of tasks that were previously considered impossible for humans. This includes tasks such as facial recognition, speech recognition, and machine learning algorithms that can analyze complex and unstructured data.
    
    2. Integration with other technologies: AI is already being integrated with other technologies such as smart homes, self-driving cars, and virtual and augmented reality. As these technologies continue to advance, it is likely that we will see


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

    insert

     occupation

     here

    ]

     by

     profession

    .

     I

     have

     a

     unique

     blend

     of

     knowledge

     and

     personality

     that

     sets

     me

     apart

     from

     the

     rest

    .

     I

    'm

     constantly

     learning

     and

     growing

    ,

     always

     seeking

     new

     experiences

     and

     new

     ways

     to

     improve

     myself

    .

     Whether

     it

    's

     teaching

    ,

     mentoring

    ,

     or

     solving

     complex

     problems

    ,

     I

    'm

     always

     willing

     to

     step

     up

     and

     lend

     a

     helping

     hand

    .

     And

     who

     knows

    ,

     maybe

     one

     day

     I

    'll

     win

     the

     lottery

    ,

     too

    .

     How

     about

     you

    ?

     What

     makes

     you

     unique

    ?

     What

     are

     your

     strengths

     and

     weaknesses

    ?

     What

    's

     your

     favorite

     hobby

     or

     interest

    ?

     And

     what

    's

     the

     most

     exciting

     thing

     you

    've

     accomplished

     so

     far

    ?

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     The

     statement

     is

    :

     "

    Paris

     is

     the

     capital

     of

     France

    ."

     This

     can

     be

     rest

    ated

     in

     a

     more

     formal

     manner

    :

     "

    The

     city

     of

     Paris

     holds

     the

     position

     of

     the

     capital

     of

     France

    ."

     
    


    Alternatively

    ,

     an

     expanded

     statement

     could

     be

    :

     "

    Paris

     is

     the

     urban

     and

     cultural

     center

     of

     France

    ,

     serving

     as

     the

     seat

     of

     the

     nation

    's

     government

    ,

     government

     representatives

    ,

     and

     the

     highest

     and

     lowest

     branches

     of

     the

     French

     state

    ,

     including

     the

     President

     of

     the

     Senate

     and

     the

     President

     of

     the

     Council

     of

     Ministers

    ."

     
    


    Each

     of

     these

     expanded

     statements

     encaps

    ulates

     the

     core

     information

     about

     Paris

    's

     importance

     in

     French

     politics

     and

     culture

    .

     The

     first

     is

     concise

     and

     directly

     states

     the

     capital

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

     and

     exciting

    .

     Some

     potential

     trends

     that

     may

     shape

     the

     AI

     landscape

     include

    :
    


    1

    .

     Self

    -learning

     systems

    :

     As

     AI

     continues

     to

     improve

    ,

     self

    -learning

     systems

     will

     become

     more

     common

    .

     These

     systems

     will

     be

     able

     to

     learn

     from

     experience

    ,

     make

     decisions

    ,

     and

     adapt

     to

     new

     information

    .
    


    2

    .

     Neural

     networks

    :

     Neural

     networks

     will

     continue

     to

     evolve

    ,

     becoming

     more

     complex

     and

     powerful

    .

     These

     networks

     will

     be

     able

     to

     process

     large

     amounts

     of

     data

     and

     make

     predictions

     based

     on

     patterns

    .
    


    3

    .

     Quantum

     AI

    :

     As

     quantum

     mechanics

     advances

    ,

     we

     may

     see

     significant

     improvements

     in

     AI

     systems

    .

     Quantum

     algorithms

     will

     be

     able

     to

     solve

     problems

     that

     are

     currently

     impossible

     for

     classical

     computers

    .
    


    4

    .

    



```python
llm.shutdown()
```
