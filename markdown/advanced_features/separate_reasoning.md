# Reasoning Parser

SGLang supports parsing reasoning content out from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

## Supported Models & Parsers

| Model  |  Reasoning tags      | Parser | Notes |
|---------|-----------------------------|------------------|-------|
| [DeepSeek‑R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `<think>` … `</think>` | `deepseek-r1` | Supports all variants (R1, R1-0528, R1-Distill) |
| [DeepSeek‑V3 series](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | `<think>` … `</think>` | `deepseek-v3` | Including [DeepSeek‑V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Supports `thinking` parameter |
| [Standard Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `<think>` … `</think>` | `qwen3` | Supports `enable_thinking` parameter |
| [Qwen3-Thinking models](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) | `<think>` … `</think>` | `qwen3` or `qwen3-thinking` | Always generates thinking content |
| [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | `◁think▷` … `◁/think▷` | `kimi_k2` | Uses special thinking delimiters. Also requires `--tool-call-parser kimi_k2` for tool use. |
| [GPT OSS](https://huggingface.co/openai/gpt-oss-120b) | `<\|channel\|>analysis<\|message\|>` … `<\|end\|>` | `gpt-oss` | N/A |
### Model-Specific Behaviors

**DeepSeek-R1 Family:**
- DeepSeek-R1: No `<think>` start tag, jumps directly to thinking content
- DeepSeek-R1-0528: Generates both `<think>` start and `</think>` end tags
- Both are handled by the same `deepseek-r1` parser

**DeepSeek-V3 Family:**
- DeepSeek-V3.1/V3.2: Hybrid model supporting both thinking and non-thinking modes, use the `deepseek-v3` parser and `thinking` parameter (NOTE: not `enable_thinking`)

**Qwen3 Family:**
- Standard Qwen3 (e.g., Qwen3-2507): Use `qwen3` parser, supports `enable_thinking` in chat templates
- Qwen3-Thinking (e.g., Qwen3-235B-A22B-Thinking-2507): Use `qwen3` or `qwen3-thinking` parser, always thinks

**Kimi K2:**
- Kimi K2 Thinking: Uses special `◁think▷` and `◁/think▷` tags. For agentic tool use, also specify `--tool-call-parser kimi_k2`.

**GPT OSS:**
- GPT OSS: Uses special `<|channel|>analysis<|message|>` and `<|end|>` tags

## Usage

### Launching the Server

Specify the `--reasoning-parser` option.


```python
import requests
from openai import OpenAI
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:02<00:02,  2.08s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.82s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:03<00:00,  1.86s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.92s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:28,  1.61s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:28,  1.61s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:05,  1.21s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:05,  1.21s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:51,  1.04it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:51,  1.04it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:42,  1.23it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:42,  1.23it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:35,  1.45it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:35,  1.45it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:29,  1.67it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:29,  1.67it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:25,  1.92it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:25,  1.92it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:21,  2.19it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:21,  2.19it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:19,  2.44it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:19,  2.44it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:16,  2.71it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:16,  2.71it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:14,  3.00it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:14,  3.00it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:13,  3.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:13,  3.32it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.68it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.68it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:10,  4.08it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:10,  4.08it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:08,  4.58it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:08,  4.58it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:11<00:07,  5.09it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:11<00:07,  5.09it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:11<00:06,  5.72it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:11<00:06,  5.72it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:11<00:05,  6.41it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:11<00:05,  6.41it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:11<00:05,  6.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:11<00:04,  7.93it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:11<00:04,  7.93it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:11<00:04,  7.93it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:11<00:03,  9.44it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:11<00:03,  9.44it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:11<00:03,  9.44it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:11<00:02, 10.97it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:11<00:02, 10.97it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:11<00:02, 10.97it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:11<00:02, 12.71it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:12<00:02, 12.71it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:12<00:01, 15.38it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:12<00:01, 15.38it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:12<00:01, 15.38it/s]

    Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:12<00:01, 15.38it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:12<00:01, 17.55it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:12<00:01, 17.55it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:12<00:01, 17.55it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:12<00:01, 17.55it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:12<00:01, 20.28it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:12<00:01, 20.28it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:12<00:01, 20.28it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:12<00:01, 20.28it/s]

    Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:12<00:01, 20.28it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 23.82it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 23.82it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:12<00:00, 23.82it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:12<00:00, 23.82it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:12<00:00, 23.82it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:12<00:00, 26.49it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:12<00:00, 26.49it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:12<00:00, 26.49it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:12<00:00, 26.49it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:12<00:00, 26.49it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 28.31it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 28.31it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 28.31it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:12<00:00, 28.31it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:12<00:00, 28.31it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:12<00:00, 30.43it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=24.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=24.81 GB):   2%|▏         | 1/58 [00:00<00:54,  1.04it/s]Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   2%|▏         | 1/58 [00:00<00:54,  1.04it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=24.71 GB):   3%|▎         | 2/58 [00:01<00:51,  1.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   3%|▎         | 2/58 [00:01<00:51,  1.10it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=24.84 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]Capturing num tokens (num_tokens=6656 avail_mem=24.97 GB):   5%|▌         | 3/58 [00:02<00:46,  1.19it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=24.97 GB):   7%|▋         | 4/58 [00:03<00:42,  1.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.11 GB):   7%|▋         | 4/58 [00:03<00:42,  1.27it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.11 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):   9%|▊         | 5/58 [00:03<00:38,  1.36it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.23 GB):  10%|█         | 6/58 [00:04<00:35,  1.48it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  10%|█         | 6/58 [00:04<00:35,  1.48it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.36 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.62it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  12%|█▏        | 7/58 [00:04<00:31,  1.62it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.09 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.16 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  16%|█▌        | 9/58 [00:05<00:24,  1.98it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.18 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.16it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.21 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=26.23 GB):  19%|█▉        | 11/58 [00:06<00:19,  2.37it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=26.23 GB):  21%|██        | 12/58 [00:06<00:17,  2.67it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  21%|██        | 12/58 [00:06<00:17,  2.67it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.26 GB):  22%|██▏       | 13/58 [00:07<00:17,  2.63it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.29 GB):  22%|██▏       | 13/58 [00:07<00:17,  2.63it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.29 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.94it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.35 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.94it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.35 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.38 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.38 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.66it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=26.40 GB):  29%|██▉       | 17/58 [00:08<00:09,  4.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.42 GB):  29%|██▉       | 17/58 [00:08<00:09,  4.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.42 GB):  31%|███       | 18/58 [00:08<00:08,  4.71it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.44 GB):  31%|███       | 18/58 [00:08<00:08,  4.71it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=26.44 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.59 GB):  33%|███▎      | 19/58 [00:08<00:07,  5.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.59 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.57 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s]

    Capturing num tokens (num_tokens=960 avail_mem=26.56 GB):  34%|███▍      | 20/58 [00:08<00:06,  6.19it/s] Capturing num tokens (num_tokens=960 avail_mem=26.56 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.64it/s]Capturing num tokens (num_tokens=896 avail_mem=26.55 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.64it/s]Capturing num tokens (num_tokens=832 avail_mem=26.54 GB):  38%|███▊      | 22/58 [00:08<00:04,  7.64it/s]

    Capturing num tokens (num_tokens=832 avail_mem=26.54 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=768 avail_mem=26.53 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  41%|████▏     | 24/58 [00:08<00:03,  8.83it/s]Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.00it/s]Capturing num tokens (num_tokens=640 avail_mem=26.51 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.00it/s]

    Capturing num tokens (num_tokens=576 avail_mem=26.50 GB):  45%|████▍     | 26/58 [00:08<00:03, 10.00it/s]Capturing num tokens (num_tokens=576 avail_mem=26.50 GB):  48%|████▊     | 28/58 [00:09<00:02, 11.19it/s]Capturing num tokens (num_tokens=512 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:09<00:02, 11.19it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  48%|████▊     | 28/58 [00:09<00:02, 11.19it/s]Capturing num tokens (num_tokens=480 avail_mem=26.48 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.25it/s]Capturing num tokens (num_tokens=448 avail_mem=26.47 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.25it/s]

    Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  52%|█████▏    | 30/58 [00:09<00:02, 12.25it/s]Capturing num tokens (num_tokens=416 avail_mem=26.45 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.13it/s]Capturing num tokens (num_tokens=384 avail_mem=26.44 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.13it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  55%|█████▌    | 32/58 [00:09<00:01, 13.13it/s]Capturing num tokens (num_tokens=352 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.48it/s]Capturing num tokens (num_tokens=320 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.48it/s]

    Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  59%|█████▊    | 34/58 [00:09<00:01, 14.48it/s]Capturing num tokens (num_tokens=288 avail_mem=26.43 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.48it/s]Capturing num tokens (num_tokens=256 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.48it/s]Capturing num tokens (num_tokens=240 avail_mem=26.41 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.48it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  62%|██████▏   | 36/58 [00:09<00:01, 15.48it/s]Capturing num tokens (num_tokens=224 avail_mem=26.40 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.21it/s]Capturing num tokens (num_tokens=208 avail_mem=26.35 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.21it/s]

    Capturing num tokens (num_tokens=192 avail_mem=26.36 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.21it/s]Capturing num tokens (num_tokens=176 avail_mem=26.37 GB):  67%|██████▋   | 39/58 [00:09<00:01, 17.21it/s]Capturing num tokens (num_tokens=176 avail_mem=26.37 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.69it/s]Capturing num tokens (num_tokens=160 avail_mem=26.33 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.69it/s]Capturing num tokens (num_tokens=144 avail_mem=26.33 GB):  72%|███████▏  | 42/58 [00:09<00:00, 18.69it/s]Capturing num tokens (num_tokens=144 avail_mem=26.33 GB):  76%|███████▌  | 44/58 [00:09<00:00, 18.94it/s]Capturing num tokens (num_tokens=128 avail_mem=26.33 GB):  76%|███████▌  | 44/58 [00:09<00:00, 18.94it/s]

    Capturing num tokens (num_tokens=112 avail_mem=26.34 GB):  76%|███████▌  | 44/58 [00:09<00:00, 18.94it/s]Capturing num tokens (num_tokens=96 avail_mem=26.30 GB):  76%|███████▌  | 44/58 [00:09<00:00, 18.94it/s] Capturing num tokens (num_tokens=96 avail_mem=26.30 GB):  81%|████████  | 47/58 [00:10<00:00, 19.39it/s]Capturing num tokens (num_tokens=80 avail_mem=26.31 GB):  81%|████████  | 47/58 [00:10<00:00, 19.39it/s]Capturing num tokens (num_tokens=64 avail_mem=26.30 GB):  81%|████████  | 47/58 [00:10<00:00, 19.39it/s]Capturing num tokens (num_tokens=48 avail_mem=26.29 GB):  81%|████████  | 47/58 [00:10<00:00, 19.39it/s]

    Capturing num tokens (num_tokens=48 avail_mem=26.29 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.95it/s]Capturing num tokens (num_tokens=32 avail_mem=26.28 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.95it/s]Capturing num tokens (num_tokens=28 avail_mem=26.27 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.95it/s]Capturing num tokens (num_tokens=24 avail_mem=26.26 GB):  86%|████████▌ | 50/58 [00:10<00:00, 19.95it/s]Capturing num tokens (num_tokens=24 avail_mem=26.26 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.73it/s]Capturing num tokens (num_tokens=20 avail_mem=26.25 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.73it/s]Capturing num tokens (num_tokens=16 avail_mem=26.24 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.73it/s]

    Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  91%|█████████▏| 53/58 [00:10<00:00, 20.73it/s]Capturing num tokens (num_tokens=12 avail_mem=26.23 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.21it/s]Capturing num tokens (num_tokens=8 avail_mem=26.22 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.21it/s] Capturing num tokens (num_tokens=4 avail_mem=26.21 GB):  97%|█████████▋| 56/58 [00:10<00:00, 21.21it/s]Capturing num tokens (num_tokens=4 avail_mem=26.21 GB): 100%|██████████| 58/58 [00:10<00:00,  5.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--reasoning-parser` defines the parser used to interpret responses.

### OpenAI Compatible API

Using the OpenAI compatible API, the contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id

messages = [
    {
        "role": "user",
        "content": "What is 1+3?",
    }
]
```

#### Non-Streaming Request


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)
print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem 1 + 3, I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together. Adding 1 and 3 gives me 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


#### Streaming Request


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>


Optionally, you can buffer the reasoning content to the last reasoning chunk (or the first chunk after the reasoning content).


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True, "stream_reasoning": False},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Finally, I conclude that the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Final Answer:** \(\boxed{4}\)</strong>


The reasoning separation is enable by default when specify . 
**To disable it, set the `separate_reasoning` option to `False` in request.**


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": False},
)

print_highlight("==== Original Output ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I identify the two numbers in the problem: 1 and 3.<br><br>Next, I add these numbers together.<br><br>Finally, I calculate the sum to find the answer.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Step 1:** Identify the numbers to add.<br>- First number: **1**<br>- Second number: **3**<br><br>**Step 2:** Add the two numbers together.<br>\[ 1 + 3 = 4 \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


### SGLang Native API 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]

print_highlight("==== Original Output ====")
print_highlight(gen_response)

parse_url = f"http://localhost:{port}/separate_reasoning"
separate_reasoning_data = {
    "text": gen_response,
    "reasoning_parser": "deepseek-r1",
}
separate_reasoning_response_json = requests.post(
    parse_url, json=separate_reasoning_data
).json()
print_highlight("==== Reasoning ====")
print_highlight(separate_reasoning_response_json["reasoning_text"])
print_highlight("==== Text ====")
print_highlight(separate_reasoning_response_json["text"])
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together to find their sum.<br><br>Finally, the result of adding 1 and 3 is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**  <br>   \[<br>   \boxed{4}<br>   \]<br><br>So, the sum of 1 and 3 is \(\boxed{4}\).</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved, which are 1 and 3.<br><br>Next, I add these two numbers together to find their sum.<br><br>Finally, the result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Final Answer:**  <br>   \[<br>   \boxed{4}<br>   \]<br><br>So, the sum of 1 and 3 is \(\boxed{4}\).</strong>



```python
terminate_process(server_process)
```

### Offline Engine API


```python
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import print_highlight

llm = sgl.Engine(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
sampling_params = {
    "max_new_tokens": 1024,
    "skip_special_tokens": False,
    "temperature": 0.6,
    "top_p": 0.95,
}
result = llm.generate(prompt=input, sampling_params=sampling_params)

generated_text = result["text"]  # Assume there is only one prompt

print_highlight("==== Original Output ====")
print_highlight(generated_text)

parser = ReasoningParser("deepseek-r1")
reasoning_text, text = parser.parse_non_stream(generated_text)
print_highlight("==== Reasoning ====")
print_highlight(reasoning_text)
print_highlight("==== Text ====")
print_highlight(text)
```

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.42s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.37s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.38s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:18,  2.48s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:18,  2.48s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:30,  1.64s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:30,  1.64s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<01:06,  1.23s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<01:06,  1.23s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:51,  1.03it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:51,  1.03it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:42,  1.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:42,  1.24it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:34,  1.48it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:34,  1.48it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:29,  1.72it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:29,  1.72it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:24,  2.03it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:24,  2.03it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:20,  2.35it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:20,  2.35it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:17,  2.67it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:17,  2.67it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:15,  3.04it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:15,  3.04it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:10<00:13,  3.42it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:10<00:13,  3.42it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:10<00:11,  3.82it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:10<00:11,  3.82it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:08,  4.75it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:08,  4.75it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:07,  5.28it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:07,  5.28it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:06,  5.86it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:06,  5.86it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:05,  6.52it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:05,  6.52it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:10<00:05,  6.52it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:11<00:04,  8.35it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:11<00:04,  8.35it/s] 

    Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:11<00:04,  8.35it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:03, 10.47it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:03, 10.47it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:03, 10.47it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:11<00:02, 12.31it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:11<00:02, 12.31it/s]

    Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:11<00:02, 12.31it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:11<00:02, 12.31it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:11<00:02, 14.30it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:11<00:02, 14.30it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:11<00:02, 14.30it/s]

    Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:11<00:01, 14.46it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:11<00:01, 14.46it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:11<00:01, 14.46it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:11<00:01, 13.95it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:11<00:01, 13.95it/s]

    Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:11<00:01, 13.95it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:11<00:01, 14.39it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:11<00:01, 14.39it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:11<00:01, 14.39it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:12<00:01, 14.74it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:12<00:01, 14.74it/s]

    Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:12<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:12<00:01, 14.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:12<00:01, 16.72it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:12<00:01, 16.72it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:12<00:01, 16.72it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:12<00:00, 17.43it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:12<00:00, 17.43it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:12<00:00, 17.43it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:12<00:00, 17.12it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:12<00:00, 17.12it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:12<00:00, 17.12it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:12<00:00, 17.10it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:12<00:00, 17.10it/s]

    Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:12<00:00, 17.10it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:12<00:00, 17.10it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:12<00:00, 17.10it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:12<00:00, 17.10it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 16.75it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 16.75it/s]

    Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 16.75it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:12<00:00, 17.00it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:12<00:00, 17.00it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:12<00:00, 17.00it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:12<00:00, 17.00it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:12<00:00, 18.94it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:12<00:00, 18.94it/s]

    Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:13<00:00, 18.94it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:13<00:00, 19.21it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:13<00:00, 19.21it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:13<00:00, 19.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:13<00:00,  4.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=25.04 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=25.04 GB):   2%|▏         | 1/58 [00:01<00:57,  1.01s/it]Capturing num tokens (num_tokens=7680 avail_mem=25.07 GB):   2%|▏         | 1/58 [00:01<00:57,  1.01s/it]

    Capturing num tokens (num_tokens=7680 avail_mem=25.07 GB):   3%|▎         | 2/58 [00:01<00:49,  1.14it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.86 GB):   3%|▎         | 2/58 [00:01<00:49,  1.14it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.86 GB):   5%|▌         | 3/58 [00:02<00:44,  1.22it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.26 GB):   5%|▌         | 3/58 [00:02<00:44,  1.22it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.26 GB):   7%|▋         | 4/58 [00:03<00:40,  1.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.86 GB):   7%|▋         | 4/58 [00:03<00:40,  1.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.86 GB):   9%|▊         | 5/58 [00:03<00:37,  1.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.86 GB):   9%|▊         | 5/58 [00:03<00:37,  1.43it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.86 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.42 GB):  10%|█         | 6/58 [00:04<00:33,  1.55it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.42 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=25.85 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=25.85 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=25.85 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.83it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=25.85 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=25.55 GB):  16%|█▌        | 9/58 [00:05<00:24,  2.00it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=25.55 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.13it/s]Capturing num tokens (num_tokens=3584 avail_mem=25.84 GB):  17%|█▋        | 10/58 [00:06<00:22,  2.13it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=25.84 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.83 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.83 GB):  21%|██        | 12/58 [00:06<00:18,  2.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=25.65 GB):  21%|██        | 12/58 [00:06<00:18,  2.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=25.65 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=25.81 GB):  22%|██▏       | 13/58 [00:07<00:16,  2.73it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=25.81 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=25.81 GB):  24%|██▍       | 14/58 [00:07<00:14,  3.00it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=25.81 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.20it/s]Capturing num tokens (num_tokens=2304 avail_mem=25.80 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.20it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=25.80 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=25.79 GB):  28%|██▊       | 16/58 [00:07<00:12,  3.49it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=25.79 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.81it/s]Capturing num tokens (num_tokens=1792 avail_mem=25.79 GB):  29%|██▉       | 17/58 [00:07<00:10,  3.81it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=25.79 GB):  31%|███       | 18/58 [00:08<00:09,  4.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.78 GB):  31%|███       | 18/58 [00:08<00:09,  4.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=25.78 GB):  33%|███▎      | 19/58 [00:08<00:08,  4.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=25.77 GB):  33%|███▎      | 19/58 [00:08<00:08,  4.40it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=25.77 GB):  34%|███▍      | 20/58 [00:08<00:07,  4.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=25.76 GB):  34%|███▍      | 20/58 [00:08<00:07,  4.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=25.76 GB):  36%|███▌      | 21/58 [00:08<00:07,  5.19it/s]Capturing num tokens (num_tokens=960 avail_mem=25.75 GB):  36%|███▌      | 21/58 [00:08<00:07,  5.19it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=25.75 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.54it/s]Capturing num tokens (num_tokens=896 avail_mem=25.74 GB):  38%|███▊      | 22/58 [00:08<00:06,  5.54it/s]Capturing num tokens (num_tokens=896 avail_mem=25.74 GB):  40%|███▉      | 23/58 [00:09<00:05,  5.86it/s]Capturing num tokens (num_tokens=832 avail_mem=25.73 GB):  40%|███▉      | 23/58 [00:09<00:05,  5.86it/s]

    Capturing num tokens (num_tokens=832 avail_mem=25.73 GB):  41%|████▏     | 24/58 [00:09<00:05,  6.08it/s]Capturing num tokens (num_tokens=768 avail_mem=25.69 GB):  41%|████▏     | 24/58 [00:09<00:05,  6.08it/s]Capturing num tokens (num_tokens=768 avail_mem=25.69 GB):  43%|████▎     | 25/58 [00:09<00:05,  6.35it/s]Capturing num tokens (num_tokens=704 avail_mem=25.72 GB):  43%|████▎     | 25/58 [00:09<00:05,  6.35it/s]

    Capturing num tokens (num_tokens=704 avail_mem=25.72 GB):  45%|████▍     | 26/58 [00:09<00:04,  6.63it/s]Capturing num tokens (num_tokens=640 avail_mem=25.72 GB):  45%|████▍     | 26/58 [00:09<00:04,  6.63it/s]Capturing num tokens (num_tokens=640 avail_mem=25.72 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.92it/s]Capturing num tokens (num_tokens=576 avail_mem=25.71 GB):  47%|████▋     | 27/58 [00:09<00:04,  6.92it/s]

    Capturing num tokens (num_tokens=576 avail_mem=25.71 GB):  48%|████▊     | 28/58 [00:09<00:03,  7.55it/s]Capturing num tokens (num_tokens=512 avail_mem=25.70 GB):  48%|████▊     | 28/58 [00:09<00:03,  7.55it/s]Capturing num tokens (num_tokens=480 avail_mem=25.69 GB):  48%|████▊     | 28/58 [00:09<00:03,  7.55it/s]Capturing num tokens (num_tokens=480 avail_mem=25.69 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.94it/s]Capturing num tokens (num_tokens=448 avail_mem=25.69 GB):  52%|█████▏    | 30/58 [00:09<00:03,  8.94it/s]

    Capturing num tokens (num_tokens=448 avail_mem=25.69 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.68it/s]Capturing num tokens (num_tokens=416 avail_mem=25.68 GB):  53%|█████▎    | 31/58 [00:09<00:03,  8.68it/s]Capturing num tokens (num_tokens=384 avail_mem=25.67 GB):  53%|█████▎    | 31/58 [00:10<00:03,  8.68it/s]

    Capturing num tokens (num_tokens=384 avail_mem=25.67 GB):  57%|█████▋    | 33/58 [00:10<00:02,  9.00it/s]Capturing num tokens (num_tokens=352 avail_mem=25.67 GB):  57%|█████▋    | 33/58 [00:10<00:02,  9.00it/s]Capturing num tokens (num_tokens=352 avail_mem=25.67 GB):  59%|█████▊    | 34/58 [00:10<00:02,  9.01it/s]Capturing num tokens (num_tokens=320 avail_mem=25.66 GB):  59%|█████▊    | 34/58 [00:10<00:02,  9.01it/s]

    Capturing num tokens (num_tokens=320 avail_mem=25.66 GB):  60%|██████    | 35/58 [00:10<00:02,  9.07it/s]Capturing num tokens (num_tokens=288 avail_mem=25.66 GB):  60%|██████    | 35/58 [00:10<00:02,  9.07it/s]Capturing num tokens (num_tokens=288 avail_mem=25.66 GB):  62%|██████▏   | 36/58 [00:10<00:02,  9.17it/s]Capturing num tokens (num_tokens=256 avail_mem=25.65 GB):  62%|██████▏   | 36/58 [00:10<00:02,  9.17it/s]

    Capturing num tokens (num_tokens=256 avail_mem=25.65 GB):  64%|██████▍   | 37/58 [00:10<00:02,  9.36it/s]Capturing num tokens (num_tokens=240 avail_mem=25.64 GB):  64%|██████▍   | 37/58 [00:10<00:02,  9.36it/s]Capturing num tokens (num_tokens=224 avail_mem=25.64 GB):  64%|██████▍   | 37/58 [00:10<00:02,  9.36it/s]Capturing num tokens (num_tokens=224 avail_mem=25.64 GB):  67%|██████▋   | 39/58 [00:10<00:01,  9.88it/s]Capturing num tokens (num_tokens=208 avail_mem=25.63 GB):  67%|██████▋   | 39/58 [00:10<00:01,  9.88it/s]

    Capturing num tokens (num_tokens=192 avail_mem=25.63 GB):  67%|██████▋   | 39/58 [00:10<00:01,  9.88it/s]Capturing num tokens (num_tokens=192 avail_mem=25.63 GB):  71%|███████   | 41/58 [00:10<00:01,  9.95it/s]Capturing num tokens (num_tokens=176 avail_mem=25.62 GB):  71%|███████   | 41/58 [00:10<00:01,  9.95it/s]Capturing num tokens (num_tokens=160 avail_mem=25.61 GB):  71%|███████   | 41/58 [00:11<00:01,  9.95it/s]

    Capturing num tokens (num_tokens=160 avail_mem=25.61 GB):  74%|███████▍  | 43/58 [00:11<00:01, 10.05it/s]Capturing num tokens (num_tokens=144 avail_mem=25.60 GB):  74%|███████▍  | 43/58 [00:11<00:01, 10.05it/s]Capturing num tokens (num_tokens=128 avail_mem=25.60 GB):  74%|███████▍  | 43/58 [00:11<00:01, 10.05it/s]Capturing num tokens (num_tokens=128 avail_mem=25.60 GB):  78%|███████▊  | 45/58 [00:11<00:01, 10.25it/s]Capturing num tokens (num_tokens=112 avail_mem=25.59 GB):  78%|███████▊  | 45/58 [00:11<00:01, 10.25it/s]

    Capturing num tokens (num_tokens=96 avail_mem=25.59 GB):  78%|███████▊  | 45/58 [00:11<00:01, 10.25it/s] Capturing num tokens (num_tokens=96 avail_mem=25.59 GB):  81%|████████  | 47/58 [00:11<00:01, 10.25it/s]Capturing num tokens (num_tokens=80 avail_mem=25.57 GB):  81%|████████  | 47/58 [00:11<00:01, 10.25it/s]Capturing num tokens (num_tokens=64 avail_mem=25.57 GB):  81%|████████  | 47/58 [00:11<00:01, 10.25it/s]

    Capturing num tokens (num_tokens=64 avail_mem=25.57 GB):  84%|████████▍ | 49/58 [00:11<00:00, 11.54it/s]Capturing num tokens (num_tokens=48 avail_mem=25.57 GB):  84%|████████▍ | 49/58 [00:11<00:00, 11.54it/s]Capturing num tokens (num_tokens=32 avail_mem=25.56 GB):  84%|████████▍ | 49/58 [00:11<00:00, 11.54it/s]Capturing num tokens (num_tokens=32 avail_mem=25.56 GB):  88%|████████▊ | 51/58 [00:11<00:00, 12.59it/s]Capturing num tokens (num_tokens=28 avail_mem=25.56 GB):  88%|████████▊ | 51/58 [00:11<00:00, 12.59it/s]Capturing num tokens (num_tokens=24 avail_mem=25.56 GB):  88%|████████▊ | 51/58 [00:11<00:00, 12.59it/s]

    Capturing num tokens (num_tokens=24 avail_mem=25.56 GB):  91%|█████████▏| 53/58 [00:11<00:00, 13.58it/s]Capturing num tokens (num_tokens=20 avail_mem=25.56 GB):  91%|█████████▏| 53/58 [00:11<00:00, 13.58it/s]Capturing num tokens (num_tokens=16 avail_mem=25.55 GB):  91%|█████████▏| 53/58 [00:11<00:00, 13.58it/s]Capturing num tokens (num_tokens=16 avail_mem=25.55 GB):  95%|█████████▍| 55/58 [00:12<00:00, 14.47it/s]Capturing num tokens (num_tokens=12 avail_mem=25.55 GB):  95%|█████████▍| 55/58 [00:12<00:00, 14.47it/s]Capturing num tokens (num_tokens=8 avail_mem=25.54 GB):  95%|█████████▍| 55/58 [00:12<00:00, 14.47it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=25.54 GB):  98%|█████████▊| 57/58 [00:12<00:00, 12.34it/s]Capturing num tokens (num_tokens=4 avail_mem=25.54 GB):  98%|█████████▊| 57/58 [00:12<00:00, 12.34it/s]Capturing num tokens (num_tokens=4 avail_mem=25.54 GB): 100%|██████████| 58/58 [00:12<00:00,  4.70it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to find the result of 1 plus 3.<br></think><br><br>To solve the problem \(1 + 3\), follow these easy steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum to find the result of 1 plus 3.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>To solve the problem \(1 + 3\), follow these easy steps:<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   \[<br>   \boxed{4}<br>   \]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
