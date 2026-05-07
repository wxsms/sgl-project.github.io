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

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:01,  1.59it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.46it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.44it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.47it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.47it/s]


    2026-05-07 00:54:57,645 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 00:54:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:57,  5.22s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:07,  2.28s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:13,  1.33s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:47,  1.14it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:23,  2.17it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.84it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.84it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.58it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.44it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.13it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.69it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.69it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.69it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.29it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.29it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.29it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.19it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.19it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.19it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.19it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.44it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.44it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.44it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.44it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.44it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.79it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]

    Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.98it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:07<00:00, 34.82it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s] 

    Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:07<00:00, 43.36it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:07<00:00, 49.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00, 60.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=61.76 GB):   2%|▏         | 1/58 [00:00<00:20,  2.84it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.07 GB):   2%|▏         | 1/58 [00:00<00:20,  2.84it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:17,  3.25it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.07 GB):   3%|▎         | 2/58 [00:00<00:17,  3.25it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=61.07 GB):   5%|▌         | 3/58 [00:00<00:18,  2.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.07 GB):   5%|▌         | 3/58 [00:00<00:18,  2.97it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.07 GB):   7%|▋         | 4/58 [00:01<00:15,  3.45it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   7%|▋         | 4/58 [00:01<00:15,  3.45it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:01<00:13,  3.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.07 GB):   9%|▊         | 5/58 [00:01<00:13,  3.85it/s]Capturing num tokens (num_tokens=5632 avail_mem=61.07 GB):  10%|█         | 6/58 [00:01<00:12,  4.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):  10%|█         | 6/58 [00:01<00:12,  4.33it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=61.07 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.07 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=61.07 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.32it/s]Capturing num tokens (num_tokens=4096 avail_mem=61.07 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.32it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=61.07 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.07 GB):  16%|█▌        | 9/58 [00:02<00:08,  5.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=61.07 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.32it/s]Capturing num tokens (num_tokens=3584 avail_mem=61.07 GB):  17%|█▋        | 10/58 [00:02<00:07,  6.32it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=61.07 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.07 GB):  19%|█▉        | 11/58 [00:02<00:06,  6.89it/s]Capturing num tokens (num_tokens=3328 avail_mem=61.07 GB):  21%|██        | 12/58 [00:02<00:06,  7.46it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.96 GB):  21%|██        | 12/58 [00:02<00:06,  7.46it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=59.96 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.96 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.96 GB):  22%|██▏       | 13/58 [00:02<00:05,  8.06it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.96 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=59.95 GB):  26%|██▌       | 15/58 [00:02<00:04,  9.46it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.95 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.95 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.95 GB):  29%|██▉       | 17/58 [00:02<00:03, 10.90it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.95 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.95 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.26it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=59.94 GB):  33%|███▎      | 19/58 [00:02<00:03, 12.26it/s]Capturing num tokens (num_tokens=960 avail_mem=59.93 GB):  33%|███▎      | 19/58 [00:03<00:03, 12.26it/s] Capturing num tokens (num_tokens=960 avail_mem=59.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=896 avail_mem=59.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=832 avail_mem=59.93 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.97it/s]Capturing num tokens (num_tokens=832 avail_mem=59.93 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.86it/s]Capturing num tokens (num_tokens=768 avail_mem=59.92 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.86it/s]

    Capturing num tokens (num_tokens=704 avail_mem=59.92 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.86it/s]Capturing num tokens (num_tokens=640 avail_mem=59.91 GB):  41%|████▏     | 24/58 [00:03<00:02, 15.86it/s]Capturing num tokens (num_tokens=640 avail_mem=59.91 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.24it/s]Capturing num tokens (num_tokens=576 avail_mem=59.87 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.24it/s]

    Capturing num tokens (num_tokens=512 avail_mem=57.84 GB):  47%|████▋     | 27/58 [00:03<00:01, 19.24it/s]Capturing num tokens (num_tokens=512 avail_mem=57.84 GB):  50%|█████     | 29/58 [00:03<00:01, 16.42it/s]Capturing num tokens (num_tokens=480 avail_mem=57.84 GB):  50%|█████     | 29/58 [00:03<00:01, 16.42it/s]Capturing num tokens (num_tokens=448 avail_mem=57.84 GB):  50%|█████     | 29/58 [00:03<00:01, 16.42it/s]Capturing num tokens (num_tokens=416 avail_mem=57.83 GB):  50%|█████     | 29/58 [00:03<00:01, 16.42it/s]Capturing num tokens (num_tokens=384 avail_mem=57.83 GB):  50%|█████     | 29/58 [00:03<00:01, 16.42it/s]Capturing num tokens (num_tokens=384 avail_mem=57.83 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.94it/s]Capturing num tokens (num_tokens=352 avail_mem=57.82 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.94it/s]Capturing num tokens (num_tokens=320 avail_mem=57.82 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.94it/s]

    Capturing num tokens (num_tokens=288 avail_mem=57.83 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.94it/s]Capturing num tokens (num_tokens=256 avail_mem=57.82 GB):  57%|█████▋    | 33/58 [00:03<00:01, 20.94it/s]Capturing num tokens (num_tokens=256 avail_mem=57.82 GB):  64%|██████▍   | 37/58 [00:03<00:00, 25.17it/s]Capturing num tokens (num_tokens=240 avail_mem=57.82 GB):  64%|██████▍   | 37/58 [00:03<00:00, 25.17it/s]Capturing num tokens (num_tokens=224 avail_mem=57.81 GB):  64%|██████▍   | 37/58 [00:03<00:00, 25.17it/s]

    Capturing num tokens (num_tokens=208 avail_mem=57.81 GB):  64%|██████▍   | 37/58 [00:10<00:00, 25.17it/s]Capturing num tokens (num_tokens=208 avail_mem=57.81 GB):  69%|██████▉   | 40/58 [00:10<00:13,  1.37it/s]Capturing num tokens (num_tokens=192 avail_mem=57.81 GB):  69%|██████▉   | 40/58 [00:10<00:13,  1.37it/s]Capturing num tokens (num_tokens=176 avail_mem=57.80 GB):  69%|██████▉   | 40/58 [00:10<00:13,  1.37it/s]Capturing num tokens (num_tokens=160 avail_mem=57.80 GB):  69%|██████▉   | 40/58 [00:10<00:13,  1.37it/s]Capturing num tokens (num_tokens=144 avail_mem=57.79 GB):  69%|██████▉   | 40/58 [00:10<00:13,  1.37it/s]Capturing num tokens (num_tokens=144 avail_mem=57.79 GB):  76%|███████▌  | 44/58 [00:11<00:06,  2.09it/s]Capturing num tokens (num_tokens=128 avail_mem=57.80 GB):  76%|███████▌  | 44/58 [00:11<00:06,  2.09it/s]Capturing num tokens (num_tokens=112 avail_mem=57.79 GB):  76%|███████▌  | 44/58 [00:11<00:06,  2.09it/s]Capturing num tokens (num_tokens=96 avail_mem=57.79 GB):  76%|███████▌  | 44/58 [00:11<00:06,  2.09it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=57.78 GB):  76%|███████▌  | 44/58 [00:11<00:06,  2.09it/s]Capturing num tokens (num_tokens=80 avail_mem=57.78 GB):  83%|████████▎ | 48/58 [00:11<00:03,  3.07it/s]Capturing num tokens (num_tokens=64 avail_mem=57.78 GB):  83%|████████▎ | 48/58 [00:11<00:03,  3.07it/s]Capturing num tokens (num_tokens=48 avail_mem=57.78 GB):  83%|████████▎ | 48/58 [00:11<00:03,  3.07it/s]Capturing num tokens (num_tokens=32 avail_mem=57.77 GB):  83%|████████▎ | 48/58 [00:11<00:03,  3.07it/s]Capturing num tokens (num_tokens=28 avail_mem=57.77 GB):  83%|████████▎ | 48/58 [00:11<00:03,  3.07it/s]Capturing num tokens (num_tokens=28 avail_mem=57.77 GB):  90%|████████▉ | 52/58 [00:11<00:01,  4.37it/s]Capturing num tokens (num_tokens=24 avail_mem=57.77 GB):  90%|████████▉ | 52/58 [00:11<00:01,  4.37it/s]Capturing num tokens (num_tokens=20 avail_mem=57.76 GB):  90%|████████▉ | 52/58 [00:11<00:01,  4.37it/s]

    Capturing num tokens (num_tokens=16 avail_mem=57.76 GB):  90%|████████▉ | 52/58 [00:11<00:01,  4.37it/s]Capturing num tokens (num_tokens=12 avail_mem=57.76 GB):  90%|████████▉ | 52/58 [00:11<00:01,  4.37it/s]Capturing num tokens (num_tokens=12 avail_mem=57.76 GB):  97%|█████████▋| 56/58 [00:11<00:00,  5.98it/s]Capturing num tokens (num_tokens=8 avail_mem=57.75 GB):  97%|█████████▋| 56/58 [00:11<00:00,  5.98it/s] Capturing num tokens (num_tokens=4 avail_mem=57.75 GB):  97%|█████████▋| 56/58 [00:11<00:00,  5.98it/s]

    Capturing num tokens (num_tokens=4 avail_mem=57.75 GB): 100%|██████████| 58/58 [00:11<00:00,  5.04it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:38294


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-07 00:55:24] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Japan** - Tokyo<br>3. **Australia** - Canberra</strong>


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


<strong style='color: #00008B;'>Sure, here are three countries along with their capitals:<br><br>1. France - Paris<br>2. Japan - Tokyo<br>3. Brazil - Brasília</strong>



<strong style='color: #00008B;'>Of course! Here is another list of three countries and their capitals:<br><br>1. Egypt - Cairo<br>2. Germany - Berlin<br>3. Australia - Canberra</strong>


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



<strong style='color: #00008B;'>2 * 2.<br><br>Let's break it down:<br>1. We have the number 2<br>2. We have the multiplication operation (*)<br>3. We have the number 2 again<br><br>So, 2 * 2 = 4<br><br>No calculator was needed in this case, as this is a simple multiplication problem.<br><br>Thus, the answer is 4.</strong>


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


<strong style='color: #00008B;'>1. **Balanced Diet**: <br>   - Focus on a variety of foods including fruits, vegetables, lean proteins, whole grains, and healthy fats.<br>   - Aim to avoid processed foods and high-sugar items.<br>   - Maintain regular meal patterns and stay hydrated by drinking plenty of water.<br><br>2. **Regular Exercise**:<br>   - Engage in at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week.<br>   - Incorporate strength training exercises at least two days per week.<br>   - Choose activities you enjoy to ensure consistency and long-term benefits.<br><br>By combining these two elements, you can significantly enhance your overall health and well-being!</strong>


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


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": "Holly",<br>        "core": "Phalaeonopsi plu",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Basilisk"<br>}</strong>


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

    100%|██████████| 3/3 [00:00<00:00, 28.90it/s]

    100%|██████████| 3/3 [00:00<00:00, 28.60it/s]

    



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
    [2026-05-07 00:55:41] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-07 00:55:45] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:02,  1.35it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.37it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.39it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:02<00:00,  1.39it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.78it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.57it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-07 00:55:56,211 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 00:55:56] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:34773



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-07 00:55:59] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


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



<strong style='color: #00008B;'>The image shows a man ironing a piece of clothing on an ironing board that is set up at the rear of a taxi in an urban environment. The taxi is a yellow cab, typical of those commonly seen in New York City, and there are other yellow taxis in the background, as well as a line of tall buildings that are characteristic of a cityscape.</strong>



```python
terminate_process(server_process)
```
