# Structured Outputs

You can specify a JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports three grammar backends:

- [XGrammar](https://github.com/mlc-ai/xgrammar)(default): Supports JSON schema, regular expression, and EBNF constraints.
- [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.
- [Llguidance](https://github.com/guidance-ai/llguidance): Supports JSON schema, regular expression, and EBNF constraints.

We suggest using XGrammar for its better performance and utility. XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). For more details, see [XGrammar technical overview](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

To use Outlines, simply add `--grammar-backend outlines` when launching the server.
To use llguidance, add `--grammar-backend llguidance`  when launching the server.
If no backend is specified, XGrammar will be used as the default.

For better output quality, **It's advisable to explicitly include instructions in the prompt to guide the model to generate the desired format.** For example, you can specify, 'Please generate the output in the following JSON format: ...'.


## OpenAI Compatible API


```python
import openai
import os

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.30it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.19it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.17it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.58it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:41,  6.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:41,  6.00s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:38,  2.84s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:38,  2.84s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:41,  1.85s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:41,  1.85s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:09,  1.29s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<00:51,  1.04it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<00:51,  1.04it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:38,  1.33it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:38,  1.33it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:30,  1.69it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:30,  1.69it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:24,  2.08it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:24,  2.08it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:09<00:19,  2.55it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:09<00:19,  2.55it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:09<00:15,  3.14it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:09<00:15,  3.14it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:12,  3.74it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:12,  3.74it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:10,  4.36it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:08,  5.10it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:08,  5.10it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:07,  5.78it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:07,  5.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:06,  6.58it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:10<00:06,  6.58it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:04,  9.09it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:04,  9.09it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:10<00:04,  9.09it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:10<00:04,  9.09it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:02, 12.98it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:02, 12.98it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:10<00:02, 12.98it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:10<00:02, 12.98it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:10<00:02, 12.98it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:10<00:01, 18.86it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:10<00:01, 26.51it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:10<00:01, 26.51it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:10<00:01, 26.51it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:10<00:01, 26.51it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:10<00:01, 24.70it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:10<00:01, 24.70it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 24.70it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 24.70it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:00, 23.53it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:10<00:00, 23.53it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:10<00:00, 26.50it/s]

    Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 29.59it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 29.59it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 29.59it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:11<00:00, 29.59it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:11<00:00, 29.59it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:11<00:00, 28.04it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:11<00:00, 28.04it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:11<00:00, 28.04it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:11<00:00, 28.04it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:11<00:00, 28.13it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:11<00:00, 31.97it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:11<00:00, 31.97it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:11<00:00, 31.97it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:11<00:00,  5.09it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=41.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=41.53 GB):   2%|▏         | 1/58 [00:00<00:40,  1.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.29 GB):   2%|▏         | 1/58 [00:00<00:40,  1.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.29 GB):   3%|▎         | 2/58 [00:01<00:37,  1.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.29 GB):   3%|▎         | 2/58 [00:01<00:37,  1.49it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.29 GB):   5%|▌         | 3/58 [00:01<00:33,  1.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.29 GB):   5%|▌         | 3/58 [00:01<00:33,  1.65it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.29 GB):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.85 GB):   7%|▋         | 4/58 [00:02<00:30,  1.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.85 GB):   9%|▊         | 5/58 [00:02<00:26,  1.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.91 GB):   9%|▊         | 5/58 [00:02<00:26,  1.99it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.91 GB):  10%|█         | 6/58 [00:03<00:23,  2.24it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.98 GB):  10%|█         | 6/58 [00:03<00:23,  2.24it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=41.98 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.48it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.27 GB):  12%|█▏        | 7/58 [00:03<00:20,  2.48it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=42.27 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.27 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.77it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.27 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.14it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.25 GB):  16%|█▌        | 9/58 [00:03<00:15,  3.14it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=42.25 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.24 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.50it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.24 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.23 GB):  19%|█▉        | 11/58 [00:04<00:12,  3.87it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.23 GB):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  21%|██        | 12/58 [00:04<00:10,  4.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.72it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.72it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.21 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.20 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.20 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.96it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.20 GB):  26%|██▌       | 15/58 [00:04<00:07,  5.96it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.20 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.18 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.18 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.18 GB):  31%|███       | 18/58 [00:05<00:05,  7.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.17 GB):  31%|███       | 18/58 [00:05<00:05,  7.92it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.17 GB):  31%|███       | 18/58 [00:05<00:05,  7.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.17 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.42it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.16 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.42it/s]Capturing num tokens (num_tokens=960 avail_mem=42.13 GB):  34%|███▍      | 20/58 [00:05<00:04,  9.42it/s] Capturing num tokens (num_tokens=960 avail_mem=42.13 GB):  38%|███▊      | 22/58 [00:05<00:03, 11.10it/s]Capturing num tokens (num_tokens=896 avail_mem=42.13 GB):  38%|███▊      | 22/58 [00:05<00:03, 11.10it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.13 GB):  38%|███▊      | 22/58 [00:05<00:03, 11.10it/s]Capturing num tokens (num_tokens=832 avail_mem=42.13 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.90it/s]Capturing num tokens (num_tokens=768 avail_mem=42.12 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.90it/s]Capturing num tokens (num_tokens=704 avail_mem=42.11 GB):  41%|████▏     | 24/58 [00:05<00:02, 11.90it/s]Capturing num tokens (num_tokens=704 avail_mem=42.11 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.42it/s]Capturing num tokens (num_tokens=640 avail_mem=42.10 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.42it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.09 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.42it/s]Capturing num tokens (num_tokens=512 avail_mem=42.08 GB):  45%|████▍     | 26/58 [00:05<00:02, 13.42it/s]Capturing num tokens (num_tokens=512 avail_mem=42.08 GB):  50%|█████     | 29/58 [00:05<00:01, 15.91it/s]Capturing num tokens (num_tokens=480 avail_mem=42.08 GB):  50%|█████     | 29/58 [00:05<00:01, 15.91it/s]Capturing num tokens (num_tokens=448 avail_mem=42.07 GB):  50%|█████     | 29/58 [00:05<00:01, 15.91it/s]Capturing num tokens (num_tokens=448 avail_mem=42.07 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.77it/s]Capturing num tokens (num_tokens=416 avail_mem=42.06 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.77it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.06 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.77it/s]Capturing num tokens (num_tokens=352 avail_mem=42.05 GB):  53%|█████▎    | 31/58 [00:06<00:01, 16.77it/s]Capturing num tokens (num_tokens=352 avail_mem=42.05 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.80it/s]Capturing num tokens (num_tokens=320 avail_mem=42.04 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.80it/s]Capturing num tokens (num_tokens=288 avail_mem=42.03 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.80it/s]Capturing num tokens (num_tokens=256 avail_mem=42.03 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.80it/s]Capturing num tokens (num_tokens=240 avail_mem=42.02 GB):  59%|█████▊    | 34/58 [00:06<00:01, 18.80it/s]

    Capturing num tokens (num_tokens=240 avail_mem=42.02 GB):  66%|██████▌   | 38/58 [00:06<00:00, 22.86it/s]Capturing num tokens (num_tokens=224 avail_mem=42.02 GB):  66%|██████▌   | 38/58 [00:06<00:00, 22.86it/s]Capturing num tokens (num_tokens=208 avail_mem=42.01 GB):  66%|██████▌   | 38/58 [00:06<00:00, 22.86it/s]Capturing num tokens (num_tokens=192 avail_mem=42.01 GB):  66%|██████▌   | 38/58 [00:06<00:00, 22.86it/s]Capturing num tokens (num_tokens=176 avail_mem=42.00 GB):  66%|██████▌   | 38/58 [00:06<00:00, 22.86it/s]Capturing num tokens (num_tokens=176 avail_mem=42.00 GB):  72%|███████▏  | 42/58 [00:06<00:00, 26.48it/s]Capturing num tokens (num_tokens=160 avail_mem=42.00 GB):  72%|███████▏  | 42/58 [00:06<00:00, 26.48it/s]Capturing num tokens (num_tokens=144 avail_mem=41.99 GB):  72%|███████▏  | 42/58 [00:06<00:00, 26.48it/s]Capturing num tokens (num_tokens=128 avail_mem=41.99 GB):  72%|███████▏  | 42/58 [00:06<00:00, 26.48it/s]Capturing num tokens (num_tokens=112 avail_mem=41.99 GB):  72%|███████▏  | 42/58 [00:06<00:00, 26.48it/s]

    Capturing num tokens (num_tokens=112 avail_mem=41.99 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.34it/s]Capturing num tokens (num_tokens=96 avail_mem=41.99 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.34it/s] Capturing num tokens (num_tokens=80 avail_mem=41.99 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.34it/s]Capturing num tokens (num_tokens=64 avail_mem=41.98 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.34it/s]Capturing num tokens (num_tokens=48 avail_mem=41.98 GB):  79%|███████▉  | 46/58 [00:06<00:00, 29.34it/s]Capturing num tokens (num_tokens=48 avail_mem=41.98 GB):  86%|████████▌ | 50/58 [00:06<00:00, 31.19it/s]Capturing num tokens (num_tokens=32 avail_mem=41.98 GB):  86%|████████▌ | 50/58 [00:06<00:00, 31.19it/s]Capturing num tokens (num_tokens=28 avail_mem=41.97 GB):  86%|████████▌ | 50/58 [00:06<00:00, 31.19it/s]Capturing num tokens (num_tokens=24 avail_mem=41.97 GB):  86%|████████▌ | 50/58 [00:06<00:00, 31.19it/s]Capturing num tokens (num_tokens=20 avail_mem=41.96 GB):  86%|████████▌ | 50/58 [00:06<00:00, 31.19it/s]

    Capturing num tokens (num_tokens=20 avail_mem=41.96 GB):  93%|█████████▎| 54/58 [00:06<00:00, 32.58it/s]Capturing num tokens (num_tokens=16 avail_mem=41.96 GB):  93%|█████████▎| 54/58 [00:06<00:00, 32.58it/s]Capturing num tokens (num_tokens=12 avail_mem=41.95 GB):  93%|█████████▎| 54/58 [00:06<00:00, 32.58it/s]Capturing num tokens (num_tokens=8 avail_mem=41.95 GB):  93%|█████████▎| 54/58 [00:06<00:00, 32.58it/s] Capturing num tokens (num_tokens=4 avail_mem=41.95 GB):  93%|█████████▎| 54/58 [00:06<00:00, 32.58it/s]Capturing num tokens (num_tokens=4 avail_mem=41.95 GB): 100%|██████████| 58/58 [00:06<00:00, 33.11it/s]Capturing num tokens (num_tokens=4 avail_mem=41.95 GB): 100%|██████████| 58/58 [00:06<00:00,  8.51it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


### JSON

you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.

**Using Pydantic**


```python
from pydantic import BaseModel, Field


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**



```python
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>{"name": "Paris", "population": 2147000}</strong>


### EBNF


```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ],
    temperature=0,
    max_tokens=32,
    extra_body={"ebnf": ebnf_grammar},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris is the capital of France</strong>


### Regular expression


```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
    max_tokens=128,
    extra_body={"regex": "(Paris|London)"},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris</strong>


### Structural Tag


```python
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
schema_get_current_date = tool_get_current_date["function"]["parameters"]


def get_messages():
    return [
        {
            "role": "system",
            "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
        },
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_current_weather>",
                "schema": schema_get_current_weather,
                "end": "</function>",
            },
            {
                "begin": "<function=get_current_date>",
                "schema": schema_get_current_date,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_weather,
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_date,
                    },
                    "end": "</function>",
                },
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Please note that the weather function call is using the default unit which is Fahrenheit. If you want the temperature in Celsius, you can change the unit to 'celsius'.</strong>


## Native API and SGLang Runtime (SRT)

### JSON

**Using Pydantic**


```python
import requests
import json
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


# Make API request
messages = [
    {
        "role": "user",
        "content": "Here is the information of the capital of France in the JSON format.\n",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json.dumps(CapitalInfo.model_json_schema()),
        },
    },
)
print_highlight(response.json())


response_data = json.loads(response.json()["text"])
# validate the response by the pydantic model
capital_info = CapitalInfo.model_validate(response_data)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '6373bd3eebe44ec7889ea84190700c22', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.15318461135029793, 'response_sent_to_client_ts': 1780037596.6241693}}</strong>



<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**


```python
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': 'ef8d204e0d0b45f88228e0b6a5691c28', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.15268171206116676, 'response_sent_to_client_ts': 1780037596.788485}}</strong>


### EBNF


```python
messages = [
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "max_new_tokens": 128,
            "temperature": 0,
            "n": 3,
            "ebnf": (
                "root ::= city | description\n"
                'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
                'description ::= city " is " status\n'
                'status ::= "the capital of " country\n'
                'country ::= "England" | "France" | "Germany" | "Italy"'
            ),
        },
        "stream": False,
        "return_logprob": False,
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '5d34a8f9b45a42c78f61138aed89e236', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1489051729440689, 'response_sent_to_client_ts': 1780037596.964997}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '952d983de90b4b1aa83f7a413ad2357d', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14883499965071678, 'response_sent_to_client_ts': 1780037596.9650102}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'c6da4a249618446dbe64e10d29031bb9', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.14879466220736504, 'response_sent_to_client_ts': 1780037596.9650147}}]</strong>


### Regular expression


```python
messages = [
    {
        "role": "user",
        "content": "Paris is the capital of",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': 'f4784526610b4eefb2055d8ca0b2c5e5', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.03118455596268177, 'response_sent_to_client_ts': 1780037597.0044546}}</strong>


### Structural Tag


```python
from transformers import AutoTokenizer

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'The capital of France.', 'output_ids': [791, 6864, 315, 9822, 13, 128009], 'meta_info': {'id': '3d69ccd498a142a5b223886dfe3db969', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 6, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13507862202823162, 'response_sent_to_client_ts': 1780037598.4150417}}</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=get_current_weather>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_weather,
                            },
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=get_current_date>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_date,
                            },
                            "end": "</function>",
                        },
                    ],
                    "at_least_one": False,
                    "stop_after_first": False,
                },
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': 'afb80c71b87b4160969921406ce7ca91', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.06016778200864792, 'response_sent_to_client_ts': 1780037598.4846566}}</strong>



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl

llm = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", grammar_backend="xgrammar"
)
```

    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.29it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.11it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.08it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.44it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.31it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:26,  5.73s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:26,  5.73s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:29,  2.68s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:29,  2.68s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:32,  1.68s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:32,  1.68s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<01:03,  1.18s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<01:03,  1.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:43,  1.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:43,  1.22it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:31,  1.66it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:23,  2.19it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:23,  2.19it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:17,  2.82it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:17,  2.82it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:13,  3.54it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:13,  3.54it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:10,  4.38it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:10,  4.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:08,  5.28it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:08,  5.28it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:08<00:08,  5.28it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:06,  6.96it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:06,  6.96it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:08<00:06,  6.96it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:05,  8.58it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:05,  8.58it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:08<00:05,  8.58it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:03, 10.47it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:03, 10.47it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:03, 10.47it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:08<00:03, 10.47it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:02, 13.68it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:02, 13.68it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:02, 13.68it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:02, 13.68it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:02, 13.68it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.09it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 26.26it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 26.26it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 26.26it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 26.26it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:09<00:01, 26.26it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:09<00:01, 26.26it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:09<00:01, 26.26it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:09<00:00, 33.94it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]

    Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:09<00:00, 43.10it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:09<00:00, 49.55it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00, 60.61it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.19it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.06 GB):   2%|▏         | 1/58 [00:00<00:23,  2.38it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   2%|▏         | 1/58 [00:00<00:23,  2.38it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:20,  2.69it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:20,  2.69it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:01<00:18,  3.01it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:01<00:18,  3.01it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   7%|▋         | 4/58 [00:01<00:17,  3.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=40.98 GB):   7%|▋         | 4/58 [00:01<00:17,  3.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=40.98 GB):   9%|▊         | 5/58 [00:01<00:15,  3.43it/s]Capturing num tokens (num_tokens=5632 avail_mem=38.20 GB):   9%|▊         | 5/58 [00:01<00:15,  3.43it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=38.20 GB):  10%|█         | 6/58 [00:01<00:13,  3.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.20 GB):  10%|█         | 6/58 [00:01<00:13,  3.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=38.20 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=38.20 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.22it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=38.20 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.20 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.66it/s]Capturing num tokens (num_tokens=4096 avail_mem=38.20 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=38.19 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.15it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=38.19 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.19 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.61it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.19 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.19 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.01it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.19 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.19 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.17it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=38.18 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.18 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.47it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.18 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.47it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.18 GB):  26%|██▌       | 15/58 [00:03<00:05,  8.47it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=38.18 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.18 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.17 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.89it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.17 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.17 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.68it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.17 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.68it/s]

    Capturing num tokens (num_tokens=960 avail_mem=38.16 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.68it/s] Capturing num tokens (num_tokens=960 avail_mem=38.16 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.80it/s]Capturing num tokens (num_tokens=896 avail_mem=38.15 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.80it/s]Capturing num tokens (num_tokens=832 avail_mem=38.15 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.80it/s]Capturing num tokens (num_tokens=768 avail_mem=38.15 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.80it/s]Capturing num tokens (num_tokens=768 avail_mem=38.15 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.40it/s]Capturing num tokens (num_tokens=704 avail_mem=38.14 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.40it/s]

    Capturing num tokens (num_tokens=640 avail_mem=38.14 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.40it/s]Capturing num tokens (num_tokens=576 avail_mem=38.13 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.40it/s]Capturing num tokens (num_tokens=576 avail_mem=38.13 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=512 avail_mem=38.13 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=480 avail_mem=38.12 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=448 avail_mem=38.12 GB):  48%|████▊     | 28/58 [00:03<00:01, 20.07it/s]Capturing num tokens (num_tokens=448 avail_mem=38.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.30it/s]Capturing num tokens (num_tokens=416 avail_mem=38.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.30it/s]

    Capturing num tokens (num_tokens=384 avail_mem=38.12 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.30it/s]Capturing num tokens (num_tokens=352 avail_mem=38.11 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.30it/s]Capturing num tokens (num_tokens=320 avail_mem=38.11 GB):  53%|█████▎    | 31/58 [00:03<00:01, 22.30it/s]Capturing num tokens (num_tokens=320 avail_mem=38.11 GB):  60%|██████    | 35/58 [00:03<00:00, 25.23it/s]Capturing num tokens (num_tokens=288 avail_mem=38.10 GB):  60%|██████    | 35/58 [00:03<00:00, 25.23it/s]Capturing num tokens (num_tokens=256 avail_mem=38.10 GB):  60%|██████    | 35/58 [00:03<00:00, 25.23it/s]Capturing num tokens (num_tokens=240 avail_mem=38.10 GB):  60%|██████    | 35/58 [00:03<00:00, 25.23it/s]Capturing num tokens (num_tokens=224 avail_mem=38.09 GB):  60%|██████    | 35/58 [00:03<00:00, 25.23it/s]

    Capturing num tokens (num_tokens=224 avail_mem=38.09 GB):  67%|██████▋   | 39/58 [00:03<00:00, 27.89it/s]Capturing num tokens (num_tokens=208 avail_mem=38.09 GB):  67%|██████▋   | 39/58 [00:03<00:00, 27.89it/s]Capturing num tokens (num_tokens=192 avail_mem=38.08 GB):  67%|██████▋   | 39/58 [00:03<00:00, 27.89it/s]Capturing num tokens (num_tokens=176 avail_mem=38.08 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.89it/s]Capturing num tokens (num_tokens=160 avail_mem=38.07 GB):  67%|██████▋   | 39/58 [00:04<00:00, 27.89it/s]Capturing num tokens (num_tokens=160 avail_mem=38.07 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.18it/s]Capturing num tokens (num_tokens=144 avail_mem=38.07 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.18it/s]Capturing num tokens (num_tokens=128 avail_mem=38.06 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.18it/s]Capturing num tokens (num_tokens=112 avail_mem=38.07 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.18it/s]Capturing num tokens (num_tokens=96 avail_mem=38.06 GB):  74%|███████▍  | 43/58 [00:04<00:00, 30.18it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 31.96it/s]Capturing num tokens (num_tokens=80 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 31.96it/s]Capturing num tokens (num_tokens=64 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 31.96it/s]Capturing num tokens (num_tokens=48 avail_mem=38.05 GB):  81%|████████  | 47/58 [00:04<00:00, 31.96it/s]Capturing num tokens (num_tokens=32 avail_mem=38.05 GB):  81%|████████  | 47/58 [00:04<00:00, 31.96it/s]Capturing num tokens (num_tokens=32 avail_mem=38.05 GB):  88%|████████▊ | 51/58 [00:04<00:00, 30.79it/s]Capturing num tokens (num_tokens=28 avail_mem=38.04 GB):  88%|████████▊ | 51/58 [00:04<00:00, 30.79it/s]Capturing num tokens (num_tokens=24 avail_mem=38.04 GB):  88%|████████▊ | 51/58 [00:04<00:00, 30.79it/s]Capturing num tokens (num_tokens=20 avail_mem=38.03 GB):  88%|████████▊ | 51/58 [00:04<00:00, 30.79it/s]

    Capturing num tokens (num_tokens=16 avail_mem=38.03 GB):  88%|████████▊ | 51/58 [00:04<00:00, 30.79it/s]Capturing num tokens (num_tokens=16 avail_mem=38.03 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.10it/s]Capturing num tokens (num_tokens=12 avail_mem=38.03 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.10it/s]Capturing num tokens (num_tokens=8 avail_mem=38.02 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.10it/s] Capturing num tokens (num_tokens=4 avail_mem=38.02 GB):  95%|█████████▍| 55/58 [00:04<00:00, 32.10it/s]Capturing num tokens (num_tokens=4 avail_mem=38.02 GB): 100%|██████████| 58/58 [00:04<00:00, 12.81it/s]


### JSON

**Using Pydantic**


```python
import json
from pydantic import BaseModel, Field

prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


sampling_params = {
    "temperature": 0.1,
    "top_p": 0.95,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}")  # validate the output by the pydantic model
    capital_info = CapitalInfo.model_validate_json(output["text"])
    print_highlight(f"Validated output: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Beijing","population":21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Paris","population":2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Dublin","population":527617}</strong>


**JSON Schema Directly**


```python
prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

sampling_params = {"temperature": 0.1, "top_p": 0.95, "json_schema": json_schema}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.<br>Generated text: {"name": "Beijing", "population": 21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.<br>Generated text: {"name": "Paris", "population": 2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.<br>Generated text: {"name": "Dublin", "population": 527617}</strong>


### EBNF



```python
prompts = [
    "Give me the information of the capital of France.",
    "Give me the information of the capital of Germany.",
    "Give me the information of the capital of Italy.",
]

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "ebnf": (
        "root ::= city | description\n"
        'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
        'description ::= city " is " status\n'
        'status ::= "the capital of " country\n'
        'country ::= "England" | "France" | "Germany" | "Italy"'
    ),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France.<br>Generated text: Paris is the capital of France</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Germany.<br>Generated text: Berlin is the capital of Germany</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: London is the capital of Italy</strong>


### Regular expression


```python
prompts = [
    "Please provide information about London as a major global city:",
    "Please provide information about Paris as a major global city:",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95, "regex": "(France|England)"}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about London as a major global city:<br>Generated text: England</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about Paris as a major global city:<br>Generated text: France</strong>


### Structural Tag


```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
prompts = [text]


sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema": schema_get_current_weather,
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": schema_get_current_date,
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_weather,
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_date,
                        },
                        "end": "</function>",
                    },
                ],
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
llm.shutdown()
```
