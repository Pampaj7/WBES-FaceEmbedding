#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing as mp
import time

# creates a profile to see where the code is spending most of its time - cool interactive
# python -m cProfile -o profile_results.prof run.py
# snakeviz profile_results.prof

# or 2 terms 1 python run.py other sudo py-spy top --pid $(pgrep -f run.py)                                   ─╯

"""

parser = argparse.ArgumentParser()
parser.add_argument('experiment_filepath', type=str, help="JSON file that contains experiment info")
parser.add_argument('data_dir', type=str, help="Path to directory that contains data")
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()
"""
experiment_filepath = "info/experiments/results_table-BFMsynth.json"
data_dir = "./data/"
num_processes = 8

avail_processes = mp.cpu_count()
"""
if args.num_processes > avail_processes:
    print(f'Warning: reducing the number of processes to {avail_processes}')
    args.num_processes = avail_processes
"""

mod = __import__("facebenchmark")

# need to add args.
with open(experiment_filepath) as f:
    experiment = json.load(f)

# report_class will be a class like resultstable, heatmapvisualizer,
# correlationreport, roicomparer -- in the json is actually resultstable
# same to reporter_class = mod.ResultsTable
reporter_class = getattr(mod, experiment['reporter_type'])
reporter_opts = experiment['reporter_opts']
reporter_cla = mod.ResultsTable

error_computers = []
for ec_json in experiment['error_computers']:
    with open(ec_json) as f:
        error_computers.append(json.load(f))

mms_info = {}
for mm in experiment['mms_info']:
    with open(experiment['mms_info'][mm]) as f:
        mms_info[mm] = json.load(f)

# need to add args
reporter = reporter_class(experiment['dataset'], error_computers,
                          experiment['rec_methods'], mms_info, data_dir,
                          num_processes=num_processes,
                          use_cache=False,
                          opts=reporter_opts)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    print("🚀 Starting execution...")
    start_time = time.time()  # ⏱ Start timing

    reporter.produce()

    end_time = time.time()  # ⏱ End timing
    elapsed_time = end_time - start_time

    print(f"✅ Execution completed in {elapsed_time:.2f} seconds ⏳")

# pyinstrument -o runtime_profile.html run.py
# runtime see
