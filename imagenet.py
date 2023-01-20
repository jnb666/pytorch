#!/usr/bin/env python
# test loading imagenet dataset
import argparse
import cProfile
import datetime
import json
import logging as log
import math
import os
import pstats
import sys
import time
from os import path

import kornia.augmentation as K
import ml
import torch

profile_limit = 20
image_size = 232

test_transform = [
    ["CenterCrop", 224],
    ["Normalize", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
]

train_transform = [
    ["RandomCrop", (224, 224)],
    ["Normalize", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
]


def elapsed(start) -> str:
    t = time.time() - start
    if t < 60:
        return f"{t:.2f}s"
    else:
        return str(datetime.timedelta(seconds=round(t)))


def benchmark(ds, total_batches, batch_size, shuffle, device, multi_process=False):
    pin_memory = not device.startswith("cuda")
    if multi_process:
        loader = ml.MultiProcessLoader(batch_size, shuffle, pin_memory=pin_memory)
    else:
        loader = ml.SingleProcessLoader(batch_size, shuffle, pin_memory=pin_memory)
    loader.start(ds)
    epoch = 1
    batch = 0
    num_batches = len(loader)
    while True:
        epoch_start = time.time()
        for i, (data, targets, id) in enumerate(loader):
            batch += 1
            sys.stdout.write(f"batch {i} / {num_batches}  \r")
            loader.release(id)
            if batch >= total_batches:
                loader.shutdown()
                return
        print(f"epoch: {epoch} elapsed = {elapsed(epoch_start)}")
        epoch += 1


def get_image(ds, index):
    print(f"== image {index} ==")
    img, tgt = ds[index]
    print(f"key={index} data={img.shape} tgt={tgt} ({ds.classes[tgt]})")

    img2 = ds.transform(img)
    print(f"transformed: data={img2.shape} shape={ds.image_shape()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=True, help="disable CUDA for evaluation")
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("--train", action="store_true", default=False, help="use training set")
    parser.add_argument("--image", type=int, default=0, help="extract given image")
    parser.add_argument("--shuffle", action="store_true", default=False, help="shuffle images")
    parser.add_argument("--convert", action="store_true", default=False, help="convert images")
    parser.add_argument("--lmdb", action="store_true", default=False, help="use LMDB database")
    parser.add_argument("--multi_process", action="store_true", default=False, help="use multi process loader")
    parser.add_argument("--batch", type=int, default=100, help="image batch size")
    parser.add_argument("--bench", type=int, default=0, help="time loading given number of batches")
    parser.add_argument("--cprofile", action="store_true", default=False, help="run python cProfile")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug, with_timestamp=False)
    device = ml.get_device(args.cpu)
    dtype = torch.uint8 if args.convert else torch.float32

    transform = ml.Transforms(train_transform) if args.train else ml.Transforms(test_transform)
    if args.lmdb:
        ds = ml.LMDBDataset("Imagenet", args.datadir, transform, image_size, train=args.train, device=device, dtype=dtype)
    else:
        ds = ml.ImagenetDataset(args.datadir, transform, image_size, train=args.train, device=device, dtype=dtype)
    print("== data ==\n" + str(ds))
    # print(ds.targets)
    print(f"len(ds) = {len(ds)}")

    if args.convert:
        ds.export_to_lmdb(args.datadir)
    elif args.bench:
        start = time.time()
        if args.cprofile:
            with cProfile.Profile() as pr:
                benchmark(ds, args.bench, args.batch, args.shuffle, device, args.multi_process)
            p = pstats.Stats(pr)
            p.sort_stats("time")
            p.print_stats(profile_limit)
            p.sort_stats("cumtime")
            p.print_stats(profile_limit)
        else:
            benchmark(ds, args.bench, args.batch, args.shuffle, device, args.multi_process)
        print(f"time to read {args.bench} batches = {elapsed(start)}")
    else:
        get_image(ds, args.image)


if __name__ == "__main__":
    main()
