#!/bin/bash
docker build -t performer-cifar10 .
docker run --gpus all -v $(pwd):/app performer-cifar10