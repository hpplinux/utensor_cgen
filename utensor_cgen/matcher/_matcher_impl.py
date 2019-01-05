from collections import deque
from itertools import product

import attr
from attr.validators import instance_of

from utensor_cgen.ir import uTensorGraph
from utensor_cgen.ir.utils import is_list_of

__all__ = ["uTensorGraphMatcher"]

@attr.s
class Association(object):
  permutations = attr.ib(validator=is_list_of(tuple))


@attr.s(frozen=True, slots=True)
class OpEqualityDelegate(object):

  # to activate all configurations
  import utensor_cgen.backend.operators

  _association_map = {}
  _compatibility_map = {}

  @attr.s
  class Equivalence(object):
    is_equal = attr.ib(validator=instance_of(bool))
    input_permutation = attr.ib(
      validator=instance_of(
        (tuple, type(None))
      )
    )

  @classmethod
  def is_associative(cls, permutations):
    def deco(op):
      if op.op_type in cls._association_map:
        raise ValueError(
          "duplicate associativity definition found for {}".format(op.op_type)
        )
      cls._association_map[op.op_type] = Association(permutations)
      return op
    return deco

  @classmethod
  def is_compatible_with(cls, other_op):
    def deco(op):
      if op.op_type not in cls._compatibility_map:
        cls._compatibility_map[op.op_type] = set()
      cls._compatibility_map[op.op_type].add(other_op.op_type)
      return op
    return deco

  @classmethod
  def query_equivalence(cls, this_op, other_op):
    """
    given two ops, return the equivalence of them
    True if they are equivalent, False o.w

    Two ops are equivelent iff
    1. the are compatible, or
    2. their op_types are the same and one of the inputs permutation agrees
    """
    # 1. compatibility
    is_compatible = cls._query_compatible(this_op, other_op)

    # 2: same op_type with agree inputs
    is_equal, input_permutation = cls._query_equal_w_association(this_op, other_op)

    return cls.Equivalence(
      is_equal=(is_compatible or is_equal),
      input_permutation=input_permutation
    )
  
  @classmethod
  def _query_compatible(cls, this_op, other_op):
    compatible_ops = cls._compatibility_map.get(this_op.op_type, set())
    return (
      this_op.op_type == other_op.op_type or
      other_op.op_type in compatible_ops
    )

  @classmethod
  def _query_equal_w_association(cls, this_op, other_op):
    association = cls._association_map.get(
      this_op.op_type,
      Association(permutations=[tuple(i for i in range(this_op.n_inputs))])
    )
    


@attr.s
class uTensorGraphMatcher(object):

  pattern_ugraph = attr.ib(validator=instance_of(uTensorGraph))
  _op_eq_delegate = attr.ib(factory=OpEqualityDelegate, init=False)

  def match_all(self, other_ugraph):
    outputs_pool = []
    for op in self.pattern_ugraph.output_ops:
      same_ops = other_ugraph.get_ops_by_type(op.op_type)
      if not same_ops:
        # there are missing output(s)
        # no way to match, return empty list
        return []
      outputs_pool.append(same_ops)
    output_candidates = product(*outputs_pool)
  
  def _visit(self, state):
    pass

@attr.s
class uTensorGraphMatch(object):

  # map from op_name to op_info
  patrn2subj_op_map = attr.ib(type=dict)
  subj2patrn_op_map = attr.ib(type=dict)
  # tensor in pattern -> tensor in target
  patrn2subj_tensor_map = attr.ib(type=dict)
  # tensor in target -> tensor in pattern
  subj2patrn_tensor_map = attr.ib(type=dict)
  pattern_ugraph = attr.ib(type=uTensorGraph)
  subject_ugraph = attr.ib(type=uTensorGraph)

  def update_op_map(self, pattern_op, subj_op):
    if pattern_op.op_type != subj_op.op_type:
      raise ValueError(
        'can not update op map with different ops: {} v.s {}'.format(
          pattern_op.op_type,
          subj_op.op_type
        )
      )
    self.patrn2subj_op_map[pattern_op.name] = subj_op
    self.subj2patrn_op_map[subj_op.name] = pattern_op
    for pattern_tensor, target_tensor in zip(pattern_op.input_tensors, subj_op.input_tensors):
      self.patrn2subj_tensor_map[pattern_tensor.name] = target_tensor
      self.subj2patrn_tensor_map[target_tensor.name] = pattern_tensor

@attr.s
class _MatchState(object):
  match = attr.ib()
  @match.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraphMatch):
      raise ValueError(
        'expecting a uTensorGraphMatch, get {}'.format(type(value))
      )
  # bfs_queue is a queue for BFS of the subject ugraph
  sub_bfs_queue = attr.ib(validator=instance_of(deque))
  # consume_queue is a queue defines the matching order of pattern ugraph
  patrn_bfs_queue = attr.ib(init=False, factory=deque)
  is_matched = attr.ib(validator=instance_of(bool), default=True)

  def __attrs_post_init__(self):
    # setup consume_queue (bfs of pattern ugraph)
    pattern_ugraph = self.match.pattern_ugraph
    bfs_queue = deque()
    bfs_queue.extend([op.name for op in pattern_ugraph.output_nodes])
    patrn_bfs_queue = deque()
    while bfs_queue: # not empty
      op_name = bfs_queue.popleft()
      patrn_bfs_queue.append(op_name)
      current_op = pattern_ugraph.ops_info[op_name]
      input_op_names = []
      for t_info in current_op.input_tensors:
        op_name = t_info.op.name
        if op_name not in input_op_names:
          input_op_names.append(op_name)
      bfs_queue.extend(input_op_names)
    self.patrn_bfs_queue = patrn_bfs_queue

  @property
  def is_done(self):
    """
    a state is done, if
    1. the patrn_bfs_queue is empty (nothing to do)
    2. there is a mismatch (is_match is False)
    """
    is_empty = len(self.patrn_bfs_queue) == 0
    return is_empty or not self.is_matched