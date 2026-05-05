# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.75it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.65it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:01<00:00,  1.55it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.51it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.55it/s]


    2026-05-05 19:44:13,418 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 19:44:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.82s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:05,  2.24s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:57,  1.07s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:46,  1.15it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:38,  1.34it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:38,  1.34it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:32,  1.55it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:32,  1.55it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:08<00:28,  1.75it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:08<00:28,  1.75it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:24,  2.00it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:24,  2.00it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:21,  2.26it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:21,  2.26it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:09<00:18,  2.49it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:09<00:18,  2.49it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:09<00:16,  2.77it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:09<00:16,  2.77it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:09<00:14,  3.05it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:09<00:14,  3.05it/s]

    Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:13,  3.32it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:13,  3.32it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:10<00:11,  3.66it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:10<00:11,  3.66it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:10<00:10,  4.07it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:10<00:10,  4.07it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:10<00:09,  4.47it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:10<00:09,  4.47it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:10<00:08,  4.98it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:10<00:08,  4.98it/s]

    Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:10<00:07,  5.47it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:10<00:07,  5.47it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:10<00:06,  6.14it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:10<00:06,  6.14it/s]

    Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:10<00:05,  6.80it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:10<00:05,  6.80it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:10<00:05,  6.80it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:11<00:04,  8.70it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:11<00:04,  8.70it/s]

    Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:11<00:04,  8.70it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:11<00:03, 10.06it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:11<00:03, 10.06it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:11<00:03, 10.06it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:11<00:02, 11.64it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:11<00:02, 11.64it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:11<00:02, 11.64it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:11<00:02, 13.30it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:11<00:02, 13.30it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:11<00:02, 13.30it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:11<00:01, 14.56it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:11<00:01, 14.56it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:11<00:01, 14.56it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:11<00:01, 15.04it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:11<00:01, 15.04it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:11<00:01, 15.04it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:11<00:01, 15.04it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:11<00:01, 16.79it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:11<00:01, 16.79it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:11<00:01, 16.79it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:11<00:01, 16.79it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:11<00:01, 16.79it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:11<00:00, 20.52it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:11<00:00, 20.52it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:12<00:00, 20.52it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:12<00:00, 20.52it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:12<00:00, 21.74it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:12<00:00, 21.74it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:12<00:00, 21.74it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:12<00:00, 21.74it/s]

    Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:12<00:00, 22.67it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:12<00:00, 22.67it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:12<00:00, 22.67it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:12<00:00, 22.67it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:12<00:00, 23.74it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:12<00:00, 23.74it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:12<00:00, 23.74it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:12<00:00, 23.74it/s]

    Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:12<00:00, 23.74it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:12<00:00, 26.09it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00, 31.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:12<00:00,  4.62it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.79 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]Capturing num tokens (num_tokens=7680 avail_mem=25.58 GB):   2%|▏         | 1/58 [00:00<00:41,  1.37it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=25.58 GB):   3%|▎         | 2/58 [00:01<00:42,  1.31it/s]Capturing num tokens (num_tokens=7168 avail_mem=25.70 GB):   3%|▎         | 2/58 [00:01<00:42,  1.31it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=25.70 GB):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=25.76 GB):   5%|▌         | 3/58 [00:02<00:42,  1.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=25.76 GB):   7%|▋         | 4/58 [00:02<00:39,  1.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=25.83 GB):   7%|▋         | 4/58 [00:02<00:39,  1.36it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=25.83 GB):   9%|▊         | 5/58 [00:03<00:37,  1.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=25.90 GB):   9%|▊         | 5/58 [00:03<00:37,  1.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=25.90 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=25.96 GB):  10%|█         | 6/58 [00:04<00:33,  1.54it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=25.96 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  12%|█▏        | 7/58 [00:04<00:30,  1.66it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=26.02 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  14%|█▍        | 8/58 [00:05<00:27,  1.80it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.96it/s]Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  16%|█▌        | 9/58 [00:05<00:25,  1.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=26.68 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=26.68 GB):  17%|█▋        | 10/58 [00:05<00:22,  2.16it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=26.68 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=25.80 GB):  19%|█▉        | 11/58 [00:06<00:20,  2.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=25.80 GB):  21%|██        | 12/58 [00:06<00:18,  2.52it/s]Capturing num tokens (num_tokens=3072 avail_mem=26.21 GB):  21%|██        | 12/58 [00:06<00:18,  2.52it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=26.21 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=26.23 GB):  22%|██▏       | 13/58 [00:06<00:16,  2.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=26.23 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.93it/s]Capturing num tokens (num_tokens=2560 avail_mem=26.26 GB):  24%|██▍       | 14/58 [00:07<00:14,  2.93it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=26.26 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=26.29 GB):  26%|██▌       | 15/58 [00:07<00:13,  3.27it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=26.29 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.31 GB):  28%|██▊       | 16/58 [00:07<00:11,  3.63it/s]Capturing num tokens (num_tokens=2048 avail_mem=26.31 GB):  29%|██▉       | 17/58 [00:07<00:10,  4.03it/s]Capturing num tokens (num_tokens=1792 avail_mem=26.64 GB):  29%|██▉       | 17/58 [00:07<00:10,  4.03it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=26.64 GB):  31%|███       | 18/58 [00:07<00:09,  4.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.63 GB):  31%|███       | 18/58 [00:07<00:09,  4.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=26.63 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=26.63 GB):  33%|███▎      | 19/58 [00:08<00:07,  4.92it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=26.63 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.60 GB):  34%|███▍      | 20/58 [00:08<00:06,  5.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=26.60 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.27it/s]Capturing num tokens (num_tokens=960 avail_mem=26.59 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.27it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=26.40 GB):  36%|███▌      | 21/58 [00:08<00:05,  6.27it/s]Capturing num tokens (num_tokens=896 avail_mem=26.40 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.80it/s]Capturing num tokens (num_tokens=832 avail_mem=26.43 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.80it/s]Capturing num tokens (num_tokens=768 avail_mem=26.42 GB):  40%|███▉      | 23/58 [00:08<00:04,  7.80it/s]

    Capturing num tokens (num_tokens=768 avail_mem=26.42 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.10it/s]Capturing num tokens (num_tokens=704 avail_mem=26.52 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.10it/s]Capturing num tokens (num_tokens=640 avail_mem=26.53 GB):  43%|████▎     | 25/58 [00:08<00:03,  9.10it/s]Capturing num tokens (num_tokens=640 avail_mem=26.53 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.09it/s]Capturing num tokens (num_tokens=576 avail_mem=26.52 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.09it/s]

    Capturing num tokens (num_tokens=512 avail_mem=26.50 GB):  47%|████▋     | 27/58 [00:08<00:03, 10.09it/s]Capturing num tokens (num_tokens=512 avail_mem=26.50 GB):  50%|█████     | 29/58 [00:08<00:02, 11.22it/s]Capturing num tokens (num_tokens=480 avail_mem=26.49 GB):  50%|█████     | 29/58 [00:08<00:02, 11.22it/s]Capturing num tokens (num_tokens=448 avail_mem=26.49 GB):  50%|█████     | 29/58 [00:09<00:02, 11.22it/s]Capturing num tokens (num_tokens=448 avail_mem=26.49 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.15it/s]Capturing num tokens (num_tokens=416 avail_mem=26.48 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.15it/s]

    Capturing num tokens (num_tokens=384 avail_mem=26.47 GB):  53%|█████▎    | 31/58 [00:09<00:02, 12.15it/s]Capturing num tokens (num_tokens=384 avail_mem=26.47 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.46it/s]Capturing num tokens (num_tokens=352 avail_mem=26.45 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.46it/s]Capturing num tokens (num_tokens=320 avail_mem=26.44 GB):  57%|█████▋    | 33/58 [00:09<00:01, 13.46it/s]Capturing num tokens (num_tokens=320 avail_mem=26.44 GB):  60%|██████    | 35/58 [00:09<00:01, 14.50it/s]Capturing num tokens (num_tokens=288 avail_mem=26.45 GB):  60%|██████    | 35/58 [00:09<00:01, 14.50it/s]

    Capturing num tokens (num_tokens=256 avail_mem=26.43 GB):  60%|██████    | 35/58 [00:09<00:01, 14.50it/s]Capturing num tokens (num_tokens=256 avail_mem=26.43 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.73it/s]Capturing num tokens (num_tokens=240 avail_mem=26.42 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.73it/s]Capturing num tokens (num_tokens=224 avail_mem=26.42 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.73it/s]Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  64%|██████▍   | 37/58 [00:09<00:01, 15.73it/s]Capturing num tokens (num_tokens=208 avail_mem=26.40 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.31it/s]Capturing num tokens (num_tokens=192 avail_mem=26.39 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.31it/s]

    Capturing num tokens (num_tokens=176 avail_mem=26.39 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.31it/s]Capturing num tokens (num_tokens=160 avail_mem=26.38 GB):  69%|██████▉   | 40/58 [00:09<00:01, 17.31it/s]Capturing num tokens (num_tokens=160 avail_mem=26.38 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.46it/s]Capturing num tokens (num_tokens=144 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.46it/s]Capturing num tokens (num_tokens=128 avail_mem=26.36 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.46it/s]Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  74%|███████▍  | 43/58 [00:09<00:00, 18.46it/s]

    Capturing num tokens (num_tokens=112 avail_mem=26.35 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.41it/s]Capturing num tokens (num_tokens=96 avail_mem=26.34 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.41it/s] Capturing num tokens (num_tokens=80 avail_mem=26.33 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.41it/s]Capturing num tokens (num_tokens=64 avail_mem=26.32 GB):  79%|███████▉  | 46/58 [00:09<00:00, 19.41it/s]Capturing num tokens (num_tokens=64 avail_mem=26.32 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.08it/s]Capturing num tokens (num_tokens=48 avail_mem=26.31 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.08it/s]Capturing num tokens (num_tokens=32 avail_mem=26.30 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.08it/s]

    Capturing num tokens (num_tokens=28 avail_mem=26.29 GB):  84%|████████▍ | 49/58 [00:10<00:00, 20.08it/s]Capturing num tokens (num_tokens=28 avail_mem=26.29 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.78it/s]Capturing num tokens (num_tokens=24 avail_mem=26.28 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.78it/s]Capturing num tokens (num_tokens=20 avail_mem=26.27 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.78it/s]Capturing num tokens (num_tokens=16 avail_mem=26.26 GB):  90%|████████▉ | 52/58 [00:10<00:00, 20.78it/s]Capturing num tokens (num_tokens=16 avail_mem=26.26 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.49it/s]Capturing num tokens (num_tokens=12 avail_mem=26.25 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.49it/s]

    Capturing num tokens (num_tokens=8 avail_mem=26.24 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.49it/s] Capturing num tokens (num_tokens=4 avail_mem=26.23 GB):  95%|█████████▍| 55/58 [00:10<00:00, 21.49it/s]Capturing num tokens (num_tokens=4 avail_mem=26.23 GB): 100%|██████████| 58/58 [00:10<00:00, 21.96it/s]Capturing num tokens (num_tokens=4 avail_mem=26.23 GB): 100%|██████████| 58/58 [00:10<00:00,  5.57it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:35895


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-05 19:44:44] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Sure, here are three countries and their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Argentina - Buenos Aires</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Certainly! Here's a list of three countries along with their capitals:<br><br>1. **Germany** - Berlin<br>2. **France** - Paris<br>3. **Japan** - Tokyo</strong>



<strong style='color: #00008B;'>Sure! Here's another list of three countries and their capitals:<br><br>1. **China** - Beijing<br>2. **Italy** - Rome<br>3. **Canada** - Ottawa</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2.<br><br>To solve this:<br><br>2 * 2 = 4<br><br>So, you don't actually need a calculator for this simple multiplication problem. The answer is 4.</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>1. **Balanced Diet**: Eating a variety of foods from all major food groups ensures you get all the necessary nutrients. Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Stay away from excessive sugars, salts, and saturated fats.<br>2. **Regular Exercise**: Engage in a mix of aerobic activities and strength training to improve your cardiovascular health, mental well-being, and physical strength. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, plus strength training on at least two days.<br><br>Together, these habits will help you maintain optimal health and improve your quality of life.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Willow",<br>        "core": "Phoenix Feather",<br>        "length": 10.75<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Nicole Kidman"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

     33%|███▎      | 1/3 [00:00<00:00,  7.43it/s]

    100%|██████████| 3/3 [00:00<00:00, 18.40it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-05 19:45:08] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-05 19:45:12] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:01<00:07,  1.83s/it]

    Multi-thread loading shards:  40% Completed | 2/5 [00:02<00:03,  1.21s/it]

    Multi-thread loading shards:  60% Completed | 3/5 [00:03<00:02,  1.01s/it]

    Multi-thread loading shards:  80% Completed | 4/5 [00:04<00:00,  1.10it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:04<00:00,  1.50it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:04<00:00,  1.14it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-05 19:45:27,629 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 19:45:27] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:33527



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-05 19:45:30] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:830: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image depicts a man standing on the trunk of a yellow taxi, with an ironing board containing clothes he is ironing. The scene appears to be on a busy street, likely in a city, with other taxis and urban activity in the background. This could be interpreted symbolically or humorously, as it is unusual and impractical to iron clothes on a taxi trunk in such an environment.</strong>



```python
terminate_process(server_process)
```
