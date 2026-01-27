# CS336 Spring 2025 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment4_data.pdf](./assignment4_data.pdf)

Details of olmo paper for refrence -> 

For details on current status and how to run refer to commands.txt

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.

- [`./cs336_data`](./cs336_data): This folder contains the main implementation completed till now
## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.