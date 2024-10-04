import collections
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
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import sympy

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton

from .scheduler import *
import graphviz


MAX_LEN = 55

BLACK = "#17202A"
RED = "#C0392B"
GREEN = "#2ECC71"
YELLOW = "#FCF3CF"
BLUE = "#2471A3"

GRAPH_OUTPUT_PATH = f"{os.environ['HOME']}/.cache/torch/inductor/ir_graph_output"


class IRPrimalNode:
    def __init__(self, primal_name):
        self.name = primal_name
        self._successors: List[IRGraphNode] = []


class IRGraphNode:
    def __init__(self, node, graph):
        self.name = node.get_name()
        self.node = node
        self.origins = []
        self.stack_traces = {}
        self.stack_trace = ''
        self._successors: List[IRGraphNode] = []
        self._predecessors: List[IRGraphNode] = []
        self.graph = graph
        
        origin_to_index = {}

        def get_order(n):
            if n not in origin_to_index:
                origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return origin_to_index[n]
        
        def trim_stack_trace(st):
            stack_trace = ''
            for line in st.split('\n'):
                if line.strip().startswith('File'):
                    for element in line.split(','):
                        _element = element.strip()
                        if _element.startswith('File'):
                            self.graph.source_code = f"{_element}"
                        if _element.startswith('line'):
                            stack_trace += f"{_element}, "
                        if _element.startswith('in'):
                            stack_trace += f"{_element}: \n"
                            break
                else: 
                    stack_trace += f"{line}\n"
            return stack_trace


        origins = [(get_order(e), e) for n in node.get_nodes() for e in n.node.origins]
        if origins:
            _origins = []
            _stack_traces = {}
            for idx, op in sorted(origins):
                if op in graph.lowered_ops and len(origins) > 1:
                    continue
                graph.lowered_ops.append(op)
                st = '<unavailable>'
                if 'stack_trace' in op.meta:
                    st = str(op.meta['stack_trace']).strip()
                elif 'val' in op.meta:
                    st = str(op.meta['val']).strip()
                _origins.append(op)
                st = trim_stack_trace(st)
                if st not in _stack_traces:
                    _stack_traces[st] = [op]
                else:
                    _stack_traces[st].append(op)
                _stack_traces[st] = _stack_traces.pop(st)
            self.node = node
            self.origins = _origins
            self.stack_traces = _stack_traces
            
        if _stack_traces.keys() and len(_stack_traces.keys()) > 0:
            self.stack_trace = trim_stack_trace(list(_stack_traces.keys())[-1])


class IRGraph:
    def __init__(self, nodes):
        self.nodes: Dict[str, IRGraphNode] = {}
        self.name_to_node = {}
        self.source_code = None
        self.lowered_ops = []

        for node in nodes:
            ir_graph_node = IRGraphNode(node, self)
            self.nodes[node.get_name()] = ir_graph_node
            
            if node.get_name():
                self.name_to_node[node.get_name()] = ir_graph_node
            for name in node.get_names():
                self.name_to_node[name] = ir_graph_node
            for write in node.read_writes.writes:
                self.name_to_node[write.name] = ir_graph_node
        
        self.create_edges()
    
    def create_edges(self):
        for ir_graph_node in self.nodes.values():
            for read in ir_graph_node.node.read_writes.reads:
                name = read.name
                if name in self.name_to_node:
                    ir_graph_node._predecessors.append(self.name_to_node[name])
                    self.name_to_node[name]._successors.append(ir_graph_node)
                else:
                    primal_node = IRPrimalNode(name)
                    ir_graph_node._predecessors.append(primal_node)

    def wrap_code(self, line, max_len=MAX_LEN):
        tokens = line.split()
        wrapped_line = ""
        current_line = ""
        for token in tokens:
            if len(current_line + token) < max_len:
                current_line += token + " "
            else:
                if wrapped_line: 
                    wrapped_line += "\\\ \\l    "
                wrapped_line += current_line.strip()
                current_line = token + " "
        
        if current_line:
            if wrapped_line: 
                wrapped_line += "\\\ \\l    "
            wrapped_line += current_line.strip()
        
        return wrapped_line

    def print_graph(self, title):
        dag = graphviz.Digraph(f"ir-{title}", comment=f"ir-{title}", filename=f"ir-{title}", format='pdf')
        dag.attr(rankdir='TB')

        for node in self.nodes.values():
            for succ in node._predecessors:
                if isinstance(succ, IRPrimalNode):
                    dag.node(succ.name, label=succ.name, shape = 'ellipse')
                    # dag.edge("<start>", succ.name)

        # stack_trace_to_node = {}
        # for node in self.nodes.values():
        #     if node.stack_trace not in stack_trace_to_node:
        #         stack_trace_to_node[node.stack_trace] = []
        #     stack_trace_to_node[node.stack_trace].append(node)

        # for key_idx, key in enumerate(stack_trace_to_node):
        #     nodes = stack_trace_to_node[key]
        #     with dag.subgraph(name=f"cluster_1_{key_idx}") as c:
        #         text = '\l'.join([self.wrap_code(line) for line in key.strip().split('\n')[-2:]])
        #         c.attr(label=f'{text}', fontname='inconsolata', color=BLUE, fontcolor=BLUE, fontsize='10')
        #         for node in nodes:
        #             c.node(node.name, label=f"{node.name}", shape='ellipse')
        
        stack_traces_to_node = {}
        for node in self.nodes.values():
            stack_traces = ','.join(node.stack_traces.keys())
            if stack_traces not in stack_traces_to_node:
                stack_traces_to_node[stack_traces] = {
                    "stack_traces": [st for st in node.stack_traces],
                    "nodes": []
                }
            stack_traces_to_node[stack_traces]["nodes"].append(node)
               
        
        for key_idx, key in enumerate(stack_traces_to_node):
            meta = stack_traces_to_node[key]
            with dag.subgraph(name=f"cluster_{key_idx}") as c:
                # c.attr(rank='same')
                text = '\l'.join([self.wrap_code(line) for line in key.strip().split('\n')[-2:]])
                c.attr(label=text, shape='rect', style='rounded,dashed', fontname='inconsolata', color=BLUE, fontcolor=BLUE, fontsize='10')
                for st in meta["stack_traces"]:
                    text = '\l'.join([self.wrap_code(line, 35) for line in st.strip().split('\n')[-2:]])
                    c.node(st, label=f"{text}", shape='rect', fontname='inconsolata', color=BLUE, fontcolor=BLUE, fontsize='10')
                for node in meta["nodes"]:
                    c.node(node.name, label=f"{node.name}", shape='ellipse')        
        
        for node in self.nodes.values():
            for succ in node._predecessors:
                if isinstance(succ, IRPrimalNode):
                    dag.edge(succ.name, node.name)

            for succ in node._successors:
                dag.edge(node.name, succ.name)

        dag.render(directory=GRAPH_OUTPUT_PATH)  

    def diff(self, other):
        result = {
            "added_nodes": [],
            "removed_nodes": [],
            "modified_nodes": [],
            "added_edges": [],
            "removed_edges": [],
            "cluster_changes": [],
        }

        current_nodes = set(self.nodes.keys())
        other_nodes = set(other.nodes.keys())

        added_nodes = other_nodes - current_nodes
        removed_nodes = current_nodes - other_nodes
        common_nodes = current_nodes & other_nodes

        result["added_nodes"].extend(added_nodes)
        result["removed_nodes"].extend(removed_nodes)

        for node_name in common_nodes:
            current_node = self.nodes[node_name]
            other_node = other.nodes[node_name]

            if len(current_node._successors) != len(other_node._successors) or \
            len(current_node._predecessors) != len(other_node._predecessors):
                result["modified_nodes"].append(node_name)

        current_clusters = collections.defaultdict(list)
        other_clusters = collections.defaultdict(list)

        for node in self.nodes.values():
            current_clusters[node.stack_trace].append(node)
        for node in other.nodes.values():
            other_clusters[node.stack_trace].append(node)

        for stack_trace, nodes in current_clusters.items():
            if stack_trace in other_clusters and len(nodes) != len(other_clusters[stack_trace]):
                result["cluster_changes"].append({
                    "stack_trace": stack_trace,
                    "change": "modified",
                    "details": f"Size changed from {len(nodes)} to {len(other_clusters[stack_trace])}"
                })

        return result
