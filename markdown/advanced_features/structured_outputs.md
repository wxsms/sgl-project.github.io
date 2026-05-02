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


    [2026-05-02 02:05:27] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-02 02:05:30] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-02 02:05:35] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-02 02:05:36] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-02 02:05:38] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-05-02 02:05:39] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.04it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:02<00:02,  1.15s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:03<00:01,  1.24s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:04<00:00,  1.01it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:04<00:00,  1.05s/it]


    2026-05-02 02:05:47,803 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 02:05:47] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:07<07:01,  7.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:07<07:01,  7.40s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:07<02:58,  3.18s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:07<02:58,  3.18s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:40,  1.83s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:40,  1.83s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:08<01:03,  1.18s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:08<01:03,  1.18s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:08<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:08<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:08<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:08<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:08<00:22,  2.22it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:08<00:22,  2.22it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:17,  2.87it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:17,  2.87it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:13,  3.56it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:13,  3.56it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:11,  4.36it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:11,  4.36it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:09<00:11,  4.36it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:07,  6.01it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:07,  6.01it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:09<00:07,  6.01it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:05,  7.51it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:05,  7.51it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:09<00:05,  7.51it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:04,  9.30it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:04,  9.30it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:09<00:04,  9.30it/s]

    Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:03, 11.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:03, 11.20it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:03, 11.20it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:09<00:03, 11.20it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:09<00:02, 14.69it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:09<00:02, 14.69it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:09<00:02, 14.69it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:09<00:02, 14.69it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:09<00:02, 14.69it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:09<00:01, 20.28it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:09<00:01, 27.51it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:09<00:00, 34.96it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:09<00:00, 34.96it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:09<00:00, 34.96it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:09<00:00, 34.96it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:09<00:00, 34.96it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:10<00:00, 34.96it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:10<00:00, 34.96it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:10<00:00, 34.96it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]

    Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:10<00:00, 42.27it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:10<00:00, 46.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00, 57.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.06 GB):   2%|▏         | 1/58 [00:00<00:19,  2.92it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   2%|▏         | 1/58 [00:00<00:19,  2.92it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.03 GB):   7%|▋         | 4/58 [00:01<00:15,  3.48it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.02 GB):   7%|▋         | 4/58 [00:01<00:15,  3.48it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.02 GB):   9%|▊         | 5/58 [00:01<00:14,  3.74it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.02 GB):   9%|▊         | 5/58 [00:01<00:14,  3.74it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.02 GB):  10%|█         | 6/58 [00:01<00:12,  4.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.03 GB):  10%|█         | 6/58 [00:01<00:12,  4.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.03 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.02 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.42it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.02 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.96 GB):  14%|█▍        | 8/58 [00:01<00:10,  4.83it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.96 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.10it/s]Capturing num tokens (num_tokens=3840 avail_mem=40.98 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.10it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=40.98 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.19 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=38.19 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.91it/s]Capturing num tokens (num_tokens=3328 avail_mem=38.19 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.91it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=38.19 GB):  21%|██        | 12/58 [00:02<00:07,  6.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.19 GB):  21%|██        | 12/58 [00:02<00:07,  6.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=38.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.13it/s]Capturing num tokens (num_tokens=2816 avail_mem=38.19 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.13it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=38.19 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=38.18 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.18 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=38.18 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=38.18 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.22it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=38.18 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=38.18 GB):  31%|███       | 18/58 [00:03<00:03, 10.98it/s]Capturing num tokens (num_tokens=1536 avail_mem=38.17 GB):  31%|███       | 18/58 [00:03<00:03, 10.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.17 GB):  31%|███       | 18/58 [00:03<00:03, 10.98it/s]Capturing num tokens (num_tokens=1280 avail_mem=38.17 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=38.17 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]

    Capturing num tokens (num_tokens=960 avail_mem=38.16 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s] Capturing num tokens (num_tokens=896 avail_mem=38.15 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]Capturing num tokens (num_tokens=896 avail_mem=38.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.01it/s]Capturing num tokens (num_tokens=832 avail_mem=38.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.01it/s]Capturing num tokens (num_tokens=768 avail_mem=38.15 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.01it/s]Capturing num tokens (num_tokens=704 avail_mem=38.14 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.01it/s]

    Capturing num tokens (num_tokens=704 avail_mem=38.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.15it/s]Capturing num tokens (num_tokens=640 avail_mem=38.14 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.15it/s]Capturing num tokens (num_tokens=576 avail_mem=38.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.15it/s]Capturing num tokens (num_tokens=512 avail_mem=38.13 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.15it/s]Capturing num tokens (num_tokens=512 avail_mem=38.13 GB):  50%|█████     | 29/58 [00:03<00:01, 19.67it/s]Capturing num tokens (num_tokens=480 avail_mem=38.12 GB):  50%|█████     | 29/58 [00:03<00:01, 19.67it/s]Capturing num tokens (num_tokens=448 avail_mem=38.12 GB):  50%|█████     | 29/58 [00:03<00:01, 19.67it/s]Capturing num tokens (num_tokens=416 avail_mem=38.12 GB):  50%|█████     | 29/58 [00:03<00:01, 19.67it/s]

    Capturing num tokens (num_tokens=416 avail_mem=38.12 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.10it/s]Capturing num tokens (num_tokens=384 avail_mem=38.12 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.10it/s]Capturing num tokens (num_tokens=352 avail_mem=38.11 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.10it/s]Capturing num tokens (num_tokens=320 avail_mem=38.11 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.10it/s]Capturing num tokens (num_tokens=288 avail_mem=38.10 GB):  55%|█████▌    | 32/58 [00:03<00:01, 22.10it/s]Capturing num tokens (num_tokens=288 avail_mem=38.10 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.88it/s]Capturing num tokens (num_tokens=256 avail_mem=38.10 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.88it/s]Capturing num tokens (num_tokens=240 avail_mem=38.09 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.88it/s]Capturing num tokens (num_tokens=224 avail_mem=38.09 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.88it/s]

    Capturing num tokens (num_tokens=208 avail_mem=38.08 GB):  62%|██████▏   | 36/58 [00:03<00:00, 24.88it/s]Capturing num tokens (num_tokens=208 avail_mem=38.08 GB):  69%|██████▉   | 40/58 [00:03<00:00, 27.41it/s]Capturing num tokens (num_tokens=192 avail_mem=38.08 GB):  69%|██████▉   | 40/58 [00:03<00:00, 27.41it/s]Capturing num tokens (num_tokens=176 avail_mem=38.08 GB):  69%|██████▉   | 40/58 [00:03<00:00, 27.41it/s]Capturing num tokens (num_tokens=160 avail_mem=38.07 GB):  69%|██████▉   | 40/58 [00:03<00:00, 27.41it/s]Capturing num tokens (num_tokens=144 avail_mem=38.07 GB):  69%|██████▉   | 40/58 [00:04<00:00, 27.41it/s]Capturing num tokens (num_tokens=144 avail_mem=38.07 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.88it/s]Capturing num tokens (num_tokens=128 avail_mem=38.06 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.88it/s]

    Capturing num tokens (num_tokens=112 avail_mem=38.07 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.88it/s]Capturing num tokens (num_tokens=96 avail_mem=38.06 GB):  76%|███████▌  | 44/58 [00:04<00:00, 27.88it/s] Capturing num tokens (num_tokens=96 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 25.71it/s]Capturing num tokens (num_tokens=80 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 25.71it/s]Capturing num tokens (num_tokens=64 avail_mem=38.06 GB):  81%|████████  | 47/58 [00:04<00:00, 25.71it/s]Capturing num tokens (num_tokens=48 avail_mem=38.05 GB):  81%|████████  | 47/58 [00:04<00:00, 25.71it/s]

    Capturing num tokens (num_tokens=48 avail_mem=38.05 GB):  86%|████████▌ | 50/58 [00:04<00:00, 25.66it/s]Capturing num tokens (num_tokens=32 avail_mem=38.05 GB):  86%|████████▌ | 50/58 [00:04<00:00, 25.66it/s]Capturing num tokens (num_tokens=28 avail_mem=38.05 GB):  86%|████████▌ | 50/58 [00:04<00:00, 25.66it/s]Capturing num tokens (num_tokens=24 avail_mem=38.02 GB):  86%|████████▌ | 50/58 [00:04<00:00, 25.66it/s]Capturing num tokens (num_tokens=24 avail_mem=38.02 GB):  91%|█████████▏| 53/58 [00:04<00:00, 23.14it/s]Capturing num tokens (num_tokens=20 avail_mem=38.02 GB):  91%|█████████▏| 53/58 [00:04<00:00, 23.14it/s]

    Capturing num tokens (num_tokens=16 avail_mem=38.01 GB):  91%|█████████▏| 53/58 [00:04<00:00, 23.14it/s]Capturing num tokens (num_tokens=12 avail_mem=38.01 GB):  91%|█████████▏| 53/58 [00:04<00:00, 23.14it/s]Capturing num tokens (num_tokens=12 avail_mem=38.01 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.47it/s]Capturing num tokens (num_tokens=8 avail_mem=37.98 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.47it/s] Capturing num tokens (num_tokens=4 avail_mem=37.41 GB):  97%|█████████▋| 56/58 [00:04<00:00, 21.47it/s]Capturing num tokens (num_tokens=4 avail_mem=37.41 GB): 100%|██████████| 58/58 [00:04<00:00, 12.30it/s]


    [2026-05-02 02:06:05] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-05-02 02:06:09] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


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


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources: <br>- get_current_date function <br>- get_current_weather function</strong>



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


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources: <br>- get_current_date function<br>- get_current_weather function</strong>


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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '74e53506736845ef8586a4d1f8806070', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.21548139117658138, 'response_sent_to_client_ts': 1777687580.9591365}}</strong>



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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '5bb98df0bcc54f98b58d99279f0434cf', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.19242326728999615, 'response_sent_to_client_ts': 1777687581.1760573}}</strong>


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


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '1dfa5bc170684c0f9368bd8a95306a13', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.10005021700635552, 'response_sent_to_client_ts': 1777687581.3162978}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'd4bf48631e1d4883b55f4d0e0813ceac', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09994362154975533, 'response_sent_to_client_ts': 1777687581.3163142}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '4e90d62d74f64eb8a45b36f7b80651a4', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.09990951418876648, 'response_sent_to_client_ts': 1777687581.3163207}}]</strong>


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


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': '74ae4e19d2be4b31b56d4c85c4efb1e9', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.037215556018054485, 'response_sent_to_client_ts': 1777687581.3637688}}</strong>


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


<strong style='color: #00008B;'>{'text': 'Paris is the capital of France.', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 13, 128009], 'meta_info': {'id': '982c67ad17e749c398cf11384d0615f0', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 8, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.21784966811537743, 'response_sent_to_client_ts': 1777687583.4202332}}</strong>



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


<strong style='color: #00008B;'>{'text': 'Paris is the capital of France.', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 13, 128009], 'meta_info': {'id': 'ee5e3cdd38944e7480b4c4ce5406a662', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 8, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11293177586048841, 'response_sent_to_client_ts': 1777687583.547772}}</strong>



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

    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:04,  1.49s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:03,  1.62s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:05<00:01,  1.72s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.28s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:05<00:00,  1.41s/it]


    2026-05-02 02:06:50,458 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-02 02:06:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:06<06:16,  6.61s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:06<06:16,  6.61s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:40,  2.87s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:40,  2.87s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:07<01:31,  1.66s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:07<01:31,  1.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:07<00:58,  1.08s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:07<00:58,  1.08s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:29,  1.78it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:29,  1.78it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:21,  2.35it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:21,  2.35it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:16,  3.00it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:16,  3.00it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:13,  3.71it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:13,  3.71it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:10,  4.58it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:10,  4.58it/s]

    Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:08<00:10,  4.58it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:07,  5.99it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:07,  5.99it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:07,  6.14it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:08<00:07,  6.14it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:05,  7.88it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:05,  7.88it/s]

    Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:08<00:05,  7.88it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:08<00:04,  9.88it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:08<00:04,  9.88it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:04,  9.88it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:08<00:04,  9.88it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:02, 13.19it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:02, 13.19it/s]

    Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:02, 13.19it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:02, 13.19it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:09<00:02, 13.19it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:09<00:01, 18.54it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]

    Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:09<00:01, 25.53it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:09<00:00, 32.82it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:09<00:00, 41.13it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s]

    Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:09<00:00, 46.76it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:09<00:00, 55.30it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:09<00:00, 55.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:09<00:00,  6.05it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=32.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=32.71 GB):   2%|▏         | 1/58 [00:00<00:20,  2.84it/s]Capturing num tokens (num_tokens=7680 avail_mem=32.67 GB):   2%|▏         | 1/58 [00:00<00:20,  2.84it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=32.67 GB):   3%|▎         | 2/58 [00:00<00:18,  2.98it/s]Capturing num tokens (num_tokens=7168 avail_mem=32.67 GB):   3%|▎         | 2/58 [00:00<00:18,  2.98it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=32.67 GB):   5%|▌         | 3/58 [00:00<00:17,  3.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=32.67 GB):   5%|▌         | 3/58 [00:00<00:17,  3.23it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=32.67 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=32.67 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=32.67 GB):   9%|▊         | 5/58 [00:01<00:14,  3.69it/s]Capturing num tokens (num_tokens=5632 avail_mem=32.67 GB):   9%|▊         | 5/58 [00:01<00:14,  3.69it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=32.67 GB):  10%|█         | 6/58 [00:01<00:13,  3.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=32.63 GB):  10%|█         | 6/58 [00:01<00:13,  3.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=32.63 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=32.62 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.28it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=32.62 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=32.62 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=32.62 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]Capturing num tokens (num_tokens=3840 avail_mem=32.62 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.02it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=32.62 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=32.62 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.41it/s]Capturing num tokens (num_tokens=3584 avail_mem=32.62 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.79it/s]Capturing num tokens (num_tokens=3328 avail_mem=32.62 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.79it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=32.62 GB):  21%|██        | 12/58 [00:02<00:08,  5.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=32.10 GB):  21%|██        | 12/58 [00:02<00:08,  5.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=32.10 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.22it/s]Capturing num tokens (num_tokens=2816 avail_mem=32.10 GB):  22%|██▏       | 13/58 [00:02<00:07,  6.22it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=32.10 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.66it/s]Capturing num tokens (num_tokens=2560 avail_mem=32.10 GB):  24%|██▍       | 14/58 [00:02<00:06,  6.66it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=32.10 GB):  26%|██▌       | 15/58 [00:03<00:07,  5.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=32.10 GB):  26%|██▌       | 15/58 [00:03<00:07,  5.82it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=32.10 GB):  28%|██▊       | 16/58 [00:03<00:07,  5.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=32.09 GB):  28%|██▊       | 16/58 [00:03<00:07,  5.31it/s]Capturing num tokens (num_tokens=2048 avail_mem=32.09 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=32.09 GB):  29%|██▉       | 17/58 [00:03<00:07,  5.26it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=32.09 GB):  31%|███       | 18/58 [00:03<00:07,  5.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=32.09 GB):  31%|███       | 18/58 [00:03<00:07,  5.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=32.09 GB):  33%|███▎      | 19/58 [00:03<00:06,  5.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=32.09 GB):  33%|███▎      | 19/58 [00:03<00:06,  5.77it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=32.09 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=32.08 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.84it/s]Capturing num tokens (num_tokens=960 avail_mem=32.07 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.84it/s] Capturing num tokens (num_tokens=960 avail_mem=32.07 GB):  38%|███▊      | 22/58 [00:04<00:04,  7.68it/s]Capturing num tokens (num_tokens=896 avail_mem=32.07 GB):  38%|███▊      | 22/58 [00:04<00:04,  7.68it/s]

    Capturing num tokens (num_tokens=832 avail_mem=32.06 GB):  38%|███▊      | 22/58 [00:04<00:04,  7.68it/s]

    Capturing num tokens (num_tokens=832 avail_mem=32.06 GB):  41%|████▏     | 24/58 [00:04<00:05,  6.51it/s]Capturing num tokens (num_tokens=768 avail_mem=55.27 GB):  41%|████▏     | 24/58 [00:04<00:05,  6.51it/s]Capturing num tokens (num_tokens=768 avail_mem=55.27 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.01it/s]Capturing num tokens (num_tokens=704 avail_mem=55.27 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.01it/s]

    Capturing num tokens (num_tokens=704 avail_mem=55.27 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.13it/s]Capturing num tokens (num_tokens=640 avail_mem=55.26 GB):  45%|████▍     | 26/58 [00:04<00:04,  7.13it/s]Capturing num tokens (num_tokens=640 avail_mem=55.26 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.06it/s]Capturing num tokens (num_tokens=576 avail_mem=55.26 GB):  47%|████▋     | 27/58 [00:04<00:04,  7.06it/s]

    Capturing num tokens (num_tokens=576 avail_mem=55.26 GB):  48%|████▊     | 28/58 [00:05<00:04,  6.94it/s]Capturing num tokens (num_tokens=512 avail_mem=55.26 GB):  48%|████▊     | 28/58 [00:05<00:04,  6.94it/s]Capturing num tokens (num_tokens=512 avail_mem=55.26 GB):  50%|█████     | 29/58 [00:05<00:03,  7.37it/s]Capturing num tokens (num_tokens=480 avail_mem=55.25 GB):  50%|█████     | 29/58 [00:05<00:03,  7.37it/s]

    Capturing num tokens (num_tokens=480 avail_mem=55.25 GB):  52%|█████▏    | 30/58 [00:05<00:03,  7.09it/s]Capturing num tokens (num_tokens=448 avail_mem=55.25 GB):  52%|█████▏    | 30/58 [00:05<00:03,  7.09it/s]Capturing num tokens (num_tokens=416 avail_mem=55.25 GB):  52%|█████▏    | 30/58 [00:05<00:03,  7.09it/s]Capturing num tokens (num_tokens=416 avail_mem=55.25 GB):  55%|█████▌    | 32/58 [00:05<00:02,  9.49it/s]Capturing num tokens (num_tokens=384 avail_mem=55.25 GB):  55%|█████▌    | 32/58 [00:05<00:02,  9.49it/s]Capturing num tokens (num_tokens=352 avail_mem=55.24 GB):  55%|█████▌    | 32/58 [00:05<00:02,  9.49it/s]

    Capturing num tokens (num_tokens=352 avail_mem=55.24 GB):  59%|█████▊    | 34/58 [00:05<00:02, 11.20it/s]Capturing num tokens (num_tokens=320 avail_mem=55.24 GB):  59%|█████▊    | 34/58 [00:05<00:02, 11.20it/s]Capturing num tokens (num_tokens=288 avail_mem=55.23 GB):  59%|█████▊    | 34/58 [00:05<00:02, 11.20it/s]Capturing num tokens (num_tokens=288 avail_mem=55.23 GB):  62%|██████▏   | 36/58 [00:05<00:01, 12.43it/s]Capturing num tokens (num_tokens=256 avail_mem=55.22 GB):  62%|██████▏   | 36/58 [00:05<00:01, 12.43it/s]Capturing num tokens (num_tokens=240 avail_mem=55.22 GB):  62%|██████▏   | 36/58 [00:05<00:01, 12.43it/s]Capturing num tokens (num_tokens=224 avail_mem=55.22 GB):  62%|██████▏   | 36/58 [00:05<00:01, 12.43it/s]

    Capturing num tokens (num_tokens=224 avail_mem=55.22 GB):  67%|██████▋   | 39/58 [00:05<00:01, 16.59it/s]Capturing num tokens (num_tokens=208 avail_mem=55.21 GB):  67%|██████▋   | 39/58 [00:05<00:01, 16.59it/s]Capturing num tokens (num_tokens=192 avail_mem=55.21 GB):  67%|██████▋   | 39/58 [00:05<00:01, 16.59it/s]Capturing num tokens (num_tokens=176 avail_mem=55.20 GB):  67%|██████▋   | 39/58 [00:05<00:01, 16.59it/s]Capturing num tokens (num_tokens=176 avail_mem=55.20 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.01it/s]Capturing num tokens (num_tokens=160 avail_mem=55.20 GB):  72%|███████▏  | 42/58 [00:05<00:00, 20.01it/s]Capturing num tokens (num_tokens=144 avail_mem=55.19 GB):  72%|███████▏  | 42/58 [00:06<00:00, 20.01it/s]Capturing num tokens (num_tokens=128 avail_mem=55.19 GB):  72%|███████▏  | 42/58 [00:06<00:00, 20.01it/s]Capturing num tokens (num_tokens=112 avail_mem=55.19 GB):  72%|███████▏  | 42/58 [00:06<00:00, 20.01it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.19 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.55it/s]Capturing num tokens (num_tokens=96 avail_mem=55.19 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.55it/s] Capturing num tokens (num_tokens=80 avail_mem=55.18 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.55it/s]Capturing num tokens (num_tokens=64 avail_mem=55.18 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.55it/s]Capturing num tokens (num_tokens=48 avail_mem=55.18 GB):  79%|███████▉  | 46/58 [00:06<00:00, 24.55it/s]Capturing num tokens (num_tokens=48 avail_mem=55.18 GB):  86%|████████▌ | 50/58 [00:06<00:00, 27.52it/s]Capturing num tokens (num_tokens=32 avail_mem=55.17 GB):  86%|████████▌ | 50/58 [00:06<00:00, 27.52it/s]Capturing num tokens (num_tokens=28 avail_mem=55.17 GB):  86%|████████▌ | 50/58 [00:06<00:00, 27.52it/s]Capturing num tokens (num_tokens=24 avail_mem=55.16 GB):  86%|████████▌ | 50/58 [00:06<00:00, 27.52it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.16 GB):  86%|████████▌ | 50/58 [00:06<00:00, 27.52it/s]Capturing num tokens (num_tokens=20 avail_mem=55.16 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.68it/s]Capturing num tokens (num_tokens=16 avail_mem=55.16 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.68it/s]Capturing num tokens (num_tokens=12 avail_mem=55.15 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.68it/s]Capturing num tokens (num_tokens=8 avail_mem=55.15 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.68it/s] Capturing num tokens (num_tokens=4 avail_mem=54.59 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.68it/s]Capturing num tokens (num_tokens=4 avail_mem=54.59 GB): 100%|██████████| 58/58 [00:06<00:00, 27.35it/s]Capturing num tokens (num_tokens=4 avail_mem=54.59 GB): 100%|██████████| 58/58 [00:06<00:00,  8.92it/s]


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
