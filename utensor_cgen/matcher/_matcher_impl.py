import attr
from attr.validators import instance_of

from itertools import product

from utensor_cgen.ir.base import uTensorGraph

__all__ = ["uTensorGraphMatcher", "uTensorGraphMatch"]


@attr.s
class uTensorGraphMatcher(object):

  pattern_ugraph = attr.ib(validator=instance_of(uTensorGraph))

  def match(self, other_ugraph):
    output_candidates = product(
      *(other_ugraph.get_ops_by_type(op.op_type)
        for op in self.pattern_ugraph.output_ops)
    )
    matched = []
  
  def _traversal(self, outupt_ops):
    pass



@attr.s
class uTensorGraphMatch(object):

  ugraph = attr.ib(validator=instance_of(uTensorGraph))
  pattern_ugraph = attr.ib(validator=instance_of(uTensorGraph))
  node_prefix = attr.ib(validator=instance_of(str))
  output_nodes = attr.ib(validator=instance_of(list))
  ops_info = attr.ib(validator=instance_of(dict))
  interio_ops = attr.ib(factory=list, init=False)

  def replace(self):
    pass
