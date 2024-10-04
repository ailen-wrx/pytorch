import collections
import copy
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
    Any,
    Counter,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import sympy

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton

from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
    cache_on_self,
    cmp,
    free_symbol_has,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    green_text,
    red_text,
    sympy_product,
)
from .virtualized import V


log = logging.getLogger(__name__)
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")

FUSION_DEBUG_PATH = f"{os.environ['HOME']}/.cache/torch/inductor"
METADATA_FILE = "fusion_metadata.csv"
FAULT_REPORT_FILE = "fusion_fault_report.csv"


class fusion_config:
    def __init__(self):
        self.start_idx = -1
        self.end_idx = -1
        self.round_idx = -1
        self.debug = False
        self.record = False
        self.fault = []

    def apply(self):
        config.record_fusion = self.record
        config.debug_fusion = self.debug
        config.fusion_round_idx = self.round_idx
        config.fusion_start_idx = self.start_idx
        config.fusion_end_idx = self.end_idx
    
    def reset(self):
        self.start_idx = -1
        self.end_idx = -1
        self.round_idx = -1
        self.debug = False
        self.record = False
        self.apply()
    
    def start_record(self):
        self.record = True
        self.apply()
    
    def start_debug(self, round_idx, start_idx, end_idx):
        self.debug = True
        self.round_idx = round_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.apply()
    
    def append_fault(self, round, idx):
        self.fault.append((round, idx))

    def start_daignose(self):
        config.fusion_fault = self.fault
    
    def stop_diagnose(self):
        config.fusion_fault = None
    
    def enable_trace(self):
        config.trace.enabled = True
    
    def disable_trace(self):
        config.trace.enabled = False


class fault_loc:
    def __init__(self, model, args, mode="brute-force", tolerant=True):
        self.model = model
        self.args = args
        self.mode = mode
        self.tolerant = tolerant
        self.metadata = []
        self.cfg = fusion_config()

    def compile(self):
        return torch.compile(copy.deepcopy(self.model), backend="inductor")

    def read_metadata(self):
        with open(f"{FUSION_DEBUG_PATH}/{METADATA_FILE}", "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.metadata.append(int(line.strip()))

    def has_fault(self):
        naive_result = self.model(*self.args)
        compiled_result = self.compile()(*self.args)
        close = torch.isclose(naive_result, compiled_result).all()
        equal = torch.equal(naive_result, compiled_result)
        if (close and self.tolerant) or equal:
            return False
        else:
            return True
        
    def record(self):
        self.cfg.start_record()
        self.compile()(*self.args)
        self.cfg.reset()
        self.read_metadata()
    
    def debug(self):
        self.record()
        for i, n in enumerate(self.metadata):
            for j in range(n):
                self.cfg.start_debug(i, j, j+1)
                has_fault = self.has_fault()
                if has_fault:
                    self.cfg.append_fault(i, j)
                self.cfg.reset()

    def diagnose(self):
        self.debug()
        self.cfg.reset()
        self.cfg.start_daignose()
        self.cfg.enable_trace()
        self.compile()(*self.args)
        self.cfg.stop_diagnose()
        self.cfg.disable_trace()

