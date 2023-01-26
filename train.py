#!/usr/bin/env python
# do command line training run
import argparse
import cProfile
import logging as log
import pstats
import sys

import ml
import torch.profiler
from torch.profiler import (ProfilerActivity, profile, schedule,
                            tensorboard_trace_handler)

profile_limit = 20


def run_cprofile(ctx):
    log.info("python cProfile enabled")
    with cProfile.Profile() as pr:
        ctx.run()
    p = pstats.Stats(pr)
    p.sort_stats("time")
    p.print_stats(profile_limit)
    p.sort_stats("cumtime")
    p.print_stats(profile_limit)


def run_torch_profile(ctx, activities):
    log.info(f"torch profiling enabled for: {activities}")

    def trace_handler(p):
        file = f"/tmp/pytorch_trace_{p.step_num}.json"
        log.info(f"write chrome trace file to {file}")
        p.export_chrome_trace(file)
        if ProfilerActivity.CPU in activities:
            output = p.key_averages().table(sort_by="cpu_time_total", row_limit=profile_limit)
            print(output)
        if ProfilerActivity.CUDA in activities:
            output = p.key_averages().table(sort_by="cuda_time_total", row_limit=profile_limit)
            print(output)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=1),
        on_trace_ready=trace_handler
    ) as p:
        ctx.run(profile=p)


def run_torch_profile_hta(ctx):
    log.info(f"HTA profiling enabled")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=1),
        on_trace_ready=tensorboard_trace_handler(dir_name="./traces", use_gzip=True),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as p:
        ctx.run(profile=p)


def run(args, ctx):
    if args.htaprofile:
        run_torch_profile_hta(ctx)
    elif args.cpuprofile or args.cudaprofile:
        activities = []
        if args.cpuprofile:
            activities.append(ProfilerActivity.CPU)
        if args.cudaprofile:
            activities.append(ProfilerActivity.CUDA)
        run_torch_profile(ctx, activities)
    elif args.cprofile:
        run_cprofile(ctx)
    else:
        ctx.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=0, help="override random seed in config if set")
    parser.add_argument("--max_items", type=int, default=0, help="limit maximum number of items per epoch")
    parser.add_argument("--max_test_items", type=int, default=0, help="limit maximum number of test items per epoch")
    parser.add_argument("--epochs", type=int, default=0, help="number of epochs to train")
    parser.add_argument("--resume", type=int, default=0, help="resume training at given epoch")
    parser.add_argument("--deterministic", action="store_true", default=False, help="deterministic runs with CUDA (slower)")
    parser.add_argument("--capture", action="store_true", default=False, help="capture model execution as a CUDA graph")
    parser.add_argument("--clear", action="store_true", default=False, help="clear data from prior runs")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("--cpuprofile", action="store_true", default=False, help="generate CPU profile")
    parser.add_argument("--cudaprofile", action="store_true", default=False, help="generate CUDA profile")
    parser.add_argument("--cprofile", action="store_true", default=False, help="run python cProfile")
    parser.add_argument("--htaprofile", action="store_true", default=False, help="collect profile trace for HTA")
    parser.add_argument("config")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device(args.cpu, args.deterministic)

    ctx = ml.CmdContext(args, device)
    try:
        run(args, ctx)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
