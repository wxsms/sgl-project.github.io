# Query VLM with Offline Engine

This tutorial demonstrates how to use SGLang's **offline Engine API** to query VLMs. We will demonstrate usage with Qwen2.5-VL and Llama 4. This section demonstrates three different calling approaches:

1. **Basic Call**: Directly pass images and text.
2. **Processor Output**: Use HuggingFace processor for data preprocessing.
3. **Precomputed Embeddings**: Pre-calculate image features to improve inference efficiency.

## Understanding the Three Input Formats

SGLang supports three ways to pass visual data, each optimized for different scenarios:

### 1. **Raw Images** - Simplest approach
- Pass PIL Images, file paths, URLs, or base64 strings directly
- SGLang handles all preprocessing automatically
- Best for: Quick prototyping, simple applications

### 2. **Processor Output** - For custom preprocessing
- Pre-process images with HuggingFace processor
- Pass the complete processor output dict with `format: "processor_output"`
- Best for: Custom image transformations, integration with existing pipelines
- Requirement: Must use `input_ids` instead of text prompt

### 3. **Precomputed Embeddings** - For maximum performance
- Pre-calculate visual embeddings using the vision encoder
- Pass embeddings with `format: "precomputed_embedding"`
- Best for: Repeated queries on same images, caching, high-throughput serving
- Performance gain: Avoids redundant vision encoder computation (30-50% speedup)

**Key Rule**: Within a single request, use only one format for all images. Don't mix formats.

The examples below demonstrate all three approaches with both Qwen2.5-VL and Llama 4 models.

## Querying Qwen2.5-VL Model


```python
import nest_asyncio

nest_asyncio.apply()

import sglang.test.doc_patch  # noqa: F401

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
chat_template = "qwen2-vl"
example_image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
```


```python
from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Generated prompt text:")
print(conv.get_prompt())
print(f"\nImage size: {image.size}")
image
```

    Generated prompt text:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What's shown here: <|vision_start|><|image_pad|><|vision_end|>?<|im_end|>
    <|im_start|>assistant
    
    
    Image size: (570, 380)





    
![png](vlm_query_files/vlm_query_4_1.png)
    



### Basic Offline Engine API Call


```python
from sglang import Engine

llm = Engine(model_path=model_path, chat_template=chat_template, log_level="warning")
```

    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [2026-06-10 05:59:39] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-06-10 05:59:42] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:00<00:00,  1.19it/s]

    Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.32it/s]Multi-thread loading shards: 100% Completed | 2/2 [00:01<00:00,  1.30it/s]


      0%|          | 0/20 [00:00<?, ?it/s]Capturing batches (bs=128 avail_mem=69.46 GB):   0%|          | 0/20 [00:00<?, ?it/s]

    Capturing batches (bs=128 avail_mem=69.46 GB):   5%|▌         | 1/20 [00:01<00:24,  1.32s/it]Capturing batches (bs=120 avail_mem=55.04 GB):   5%|▌         | 1/20 [00:01<00:24,  1.32s/it]Capturing batches (bs=112 avail_mem=55.04 GB):   5%|▌         | 1/20 [00:01<00:24,  1.32s/it]Capturing batches (bs=112 avail_mem=55.04 GB):  15%|█▌        | 3/20 [00:01<00:06,  2.51it/s]Capturing batches (bs=104 avail_mem=55.02 GB):  15%|█▌        | 3/20 [00:01<00:06,  2.51it/s]

    Capturing batches (bs=96 avail_mem=55.02 GB):  15%|█▌        | 3/20 [00:01<00:06,  2.51it/s] Capturing batches (bs=96 avail_mem=55.02 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.37it/s]Capturing batches (bs=88 avail_mem=55.02 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.37it/s]Capturing batches (bs=80 avail_mem=55.00 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.37it/s]Capturing batches (bs=72 avail_mem=54.52 GB):  25%|██▌       | 5/20 [00:01<00:03,  4.37it/s]

    Capturing batches (bs=72 avail_mem=54.52 GB):  40%|████      | 8/20 [00:01<00:01,  7.52it/s]Capturing batches (bs=64 avail_mem=54.36 GB):  40%|████      | 8/20 [00:01<00:01,  7.52it/s]Capturing batches (bs=56 avail_mem=54.36 GB):  40%|████      | 8/20 [00:01<00:01,  7.52it/s]Capturing batches (bs=48 avail_mem=54.35 GB):  40%|████      | 8/20 [00:01<00:01,  7.52it/s]Capturing batches (bs=48 avail_mem=54.35 GB):  55%|█████▌    | 11/20 [00:01<00:00, 10.57it/s]Capturing batches (bs=40 avail_mem=54.35 GB):  55%|█████▌    | 11/20 [00:01<00:00, 10.57it/s]Capturing batches (bs=32 avail_mem=54.35 GB):  55%|█████▌    | 11/20 [00:01<00:00, 10.57it/s]

    Capturing batches (bs=24 avail_mem=54.35 GB):  55%|█████▌    | 11/20 [00:01<00:00, 10.57it/s]Capturing batches (bs=24 avail_mem=54.35 GB):  70%|███████   | 14/20 [00:02<00:00, 13.30it/s]Capturing batches (bs=16 avail_mem=54.35 GB):  70%|███████   | 14/20 [00:02<00:00, 13.30it/s]Capturing batches (bs=12 avail_mem=54.35 GB):  70%|███████   | 14/20 [00:02<00:00, 13.30it/s]Capturing batches (bs=12 avail_mem=54.35 GB):  80%|████████  | 16/20 [00:02<00:00, 14.13it/s]Capturing batches (bs=8 avail_mem=54.35 GB):  80%|████████  | 16/20 [00:02<00:00, 14.13it/s] 

    Capturing batches (bs=4 avail_mem=54.34 GB):  80%|████████  | 16/20 [00:02<00:00, 14.13it/s]Capturing batches (bs=2 avail_mem=54.34 GB):  80%|████████  | 16/20 [00:02<00:00, 14.13it/s]Capturing batches (bs=2 avail_mem=54.34 GB):  95%|█████████▌| 19/20 [00:02<00:00, 16.91it/s]Capturing batches (bs=1 avail_mem=54.34 GB):  95%|█████████▌| 19/20 [00:02<00:00, 16.91it/s]Capturing batches (bs=1 avail_mem=54.34 GB): 100%|██████████| 20/20 [00:02<00:00,  8.80it/s]



```python
out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Model response:")
print(out["text"])
```

    Model response:
    The image shows a street scene with two yellow cabs in New York City. One of the cabs has a_MSJ знак staked into the vehicle. The other image is a ayrı 다에라 of one of the taxis with wrought iron straps hung on the rear side.


### Call with Processor Output

Using a HuggingFace processor to preprocess text and images, and passing the `processor_output` directly into `Engine.generate`.


```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out["text"])
```

    Response using processor output:
    The image shows a man standing on a city street, holding up laundry items to resemble a road sign. The sign reads "PARKING" in large, bold letters. This humorous scene is likely meant to convey that parking should be done in designated areas only, rather than being used for other purposes like being a more pronounceable sign for vehicles to follow.


### Call with Precomputed Embeddings

You can pre-calculate image features to avoid repeated visual encoding processes.


```python
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()
vision = model.model.visual.cuda()
```


    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/824 [00:00<?, ?it/s]



```python
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

precomputed_embeddings = vision(
    processor_output["pixel_values"].cuda(), processor_output["image_grid_thw"].cuda()
)
precomputed_embeddings = precomputed_embeddings.pooler_output

multi_modal_item = dict(
    processor_output,
    format="precomputed_embedding",
    feature=precomputed_embeddings,
)

out = llm.generate(input_ids=input_ids, image_data=[multi_modal_item])
print("Response using precomputed embeddings:")
print(out["text"])

llm.shutdown()
```

    Response using precomputed embeddings:
    The image shows a person in a yellow shirt ironing a piece of clothing on a spike behind a yellow taxi. There are also two flags in the background.


## Querying Llama 4 Vision Model

```python
model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
chat_template = "llama-4"

from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

# Download the same example image
image = Image.open(BytesIO(requests.get(example_image_url).content))

conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Llama 4 generated prompt text:")
print(conv.get_prompt())
print(f"Image size: {image.size}")

image
```

### Llama 4 Basic Call

Llama 4 requires more computational resources, so it's configured with multi-GPU parallelism (tp_size=4) and larger context length.

```python
llm = Engine(
    model_path=model_path,
    enable_multimodal=True,
    attention_backend="fa3",
    tp_size=4,
    context_length=65536,
)

out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
print("Llama 4 response:")
print(out["text"])
```

### Call with Processor Output

Using HuggingFace processor to preprocess data can reduce computational overhead during inference.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor_output = processor(
    images=[image], text=conv.get_prompt(), return_tensors="pt"
)

out = llm.generate(
    input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
    image_data=[dict(processor_output, format="processor_output")],
)
print("Response using processor output:")
print(out)
```

### Call with Precomputed Embeddings

```python
from transformers import AutoProcessor
from transformers import Llama4ForConditionalGeneration

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto"
).eval()

vision = model.vision_model.cuda()
multi_modal_projector = model.multi_modal_projector.cuda()

print(f'Image pixel values shape: {processor_output["pixel_values"].shape}')
input_ids = processor_output["input_ids"][0].detach().cpu().tolist()

# Process image through vision encoder
image_outputs = vision(
    processor_output["pixel_values"].to("cuda"), 
    aspect_ratio_ids=processor_output["aspect_ratio_ids"].to("cuda"),
    aspect_ratio_mask=processor_output["aspect_ratio_mask"].to("cuda"),
    output_hidden_states=False
)
image_features = image_outputs.last_hidden_state

# Flatten image features and pass through multimodal projector
vision_flat = image_features.view(-1, image_features.size(-1))
precomputed_embeddings = multi_modal_projector(vision_flat)

# Build precomputed embedding data item
mm_item = dict(
    processor_output, 
    format="precomputed_embedding", 
    feature=precomputed_embeddings
)

# Use precomputed embeddings for efficient inference
out = llm.generate(input_ids=input_ids, image_data=[mm_item])
print("Llama 4 precomputed embedding response:")
print(out["text"])
```
