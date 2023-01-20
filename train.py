#!/usr/bin/env python
# do command line training run
import argparse
import cProfile
import logging as log
import pstats
import sys

import ml
import torch.profiler
from torch.profiler import ProfilerActivity, profile

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
    sort_key = "cuda_time_total" if ProfilerActivity.CUDA in activities else "cpu_time_total"

    def trace_handler(p):
        file = f"/tmp/pytorch_trace_{p.step_num}.json"
        log.info(f"write trace file to {file}")
        p.export_chrome_trace(file)
        if sort_key == "cpu_time_total":
            output = p.key_averages().table(sort_by="self_" + sort_key, row_limit=profile_limit)
            print(output)
        output = p.key_averages().table(sort_by=sort_key, row_limit=profile_limit)
        print(output)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=1, active=1),
        on_trace_ready=trace_handler
    ) as p:
        ctx.run(profile=p)


def run(args, ctx):
    if args.cpuprofile or args.cudaprofile:
        if args.cpuprofile:
            activities = [ProfilerActivity.CPU]
        if args.cudaprofile:
            activities = [ProfilerActivity.CUDA]
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
    parser.add_argument("--epochs", type=int, default=0, help="number of epochs to train")
    parser.add_argument("--resume", type=int, default=0, help="resume training at given epoch")
    parser.add_argument("--deterministic", action="store_true", default=False, help="deterministic runs with CUDA (slower)")
    parser.add_argument("--clear", action="store_true", default=False, help="clear data from prior runs")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("--cpuprofile", action="store_true", default=False, help="generate CPU profile")
    parser.add_argument("--cudaprofile", action="store_true", default=False, help="generate CUDA profile")
    parser.add_argument("--cprofile", action="store_true", default=False, help="run python cProfile")
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
