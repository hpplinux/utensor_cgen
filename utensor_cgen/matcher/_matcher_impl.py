from collections import defaultdict, deque
from itertools import product

import attr
from attr.validators import instance_of

from utensor_cgen.ir import uTensorGraph
from utensor_cgen.ir.utils import is_list_of

__all__ = ["uTensorGraphMatcher"]

class OpEqualityDelegate(object):

  # to activate all configurations
  import utensor_cgen.backend.operators

  _association_map = {}
  _compatibility_map = defaultdict(set)

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
      cls._compatibility_map[op.op_type].add(other_op.op_type)
      return op
    return deco

  def visit(self, state):
    """
    query if this op info is equivelent to other op info
    in the context of the graph
    """
    target_op = state.ops_queue.popleft()
    pattern_op = state.match.inv_op_names_map[target_op.name]
    compatible_ops = self._compatibility_map.get(pattern_op.op_type, set())
    if target_op.op_type in compatible_ops:
      return True

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

    states = []
    for candidates in output_candidates:
      state = _MatchState(
        match=uTensorGraphMatch(
          pattern_ugraph=self.pattern_ugraph,
          target_ugraph=other_ugraph
        ),
        ops_queue=deque(),
        is_matched=True
      )
      for pattern_op, target_op in zip(self.pattern_ugraph.output_ops, candidates):
        state.match.update_op_map(pattern_op, target_op)
        state.enqueue(target_op)

    while True:
      if all([state.is_done for state in states]):
        break
      for state in states:
        self._op_eq_delegate.visit(state)
    return [state.match for state in states if state.is_matched]

@attr.s
class uTensorGraphMatch(object):

  # these dicts map a given name in the pattern ugraph
  # to the counter part of the target ugraph
  # op in pattern -> op in target
  op_names_map = attr.ib(type=dict)
  # op in target -> op in pattern
  inv_op_names_map = attr.ib(type=dict)
  # tensor in pattern -> tensor in target
  tensor_names_map = attr.ib(type=dict)
  # tensor in target -> tensor in pattern
  inv_tensor_names_map = attr.ib(type=dict)
  pattern_ugraph = attr.ib(type=uTensorGraph)
  target_ugraph = attr.ib(type=uTensorGraph)

  def update_op_map(self, pattern_op, target_op):
    self.op_names_map[pattern_op.name] = target_op
    self.inv_op_names_map[target_op.name] = pattern_op
    for pattern_tensor, target_tensor in zip(pattern_op.input_tensors, target_op.input_tensors):
      self.tensor_names_map[pattern_tensor.name] = target_tensor
      self.inv_tensor_names_map[target_tensor.name] = pattern_tensor

@attr.s
class _MatchState(object):
  match = attr.ib()
  @match.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraphMatch):
      raise ValueError(
        'expecting a uTensorGraphMatch, get {}'.format(type(value))
      )
  ops_queue = attr.ib(validator=instance_of(deque))
  is_matched = attr.ib(type=bool, default=True)

  @property
  def is_done(self):
    return self.ops_queue.is_empty or not self.is_matched

  def enqueue(self, op_info):
    pass


@attr.s
class Association(object):
  permutations = attr.ib(validator=is_list_of(tuple))
