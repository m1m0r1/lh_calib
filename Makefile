SHELL := /bin/bash
PYTHON = python
LH_BASE = .

# e.g. qsub
_RUN_CPU =
_RUN_GPU =
include make/*.mk
