# -*- coding:utf8 -*-
import logging
import os
import pickle
import pdb
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.tools.graph_transforms import TransformGraph

from .ir import uTensorGraph
from .operators import OperatorFactory
from .snippets import (CommentSnippet, ContextGlobalArrayContainer,
                       ContextHeaderSnippet, ContextSnippetsContainer,
                       CreateTensorBinarySnippet, CreateTensorIdxSnippet)
from .snippets.composer import Composer
from .transformer.optimizer import RefCntOptimizer
from .transformer.pipline import TransformerPipeline
from .utils import NamescopedKWArgsParser

__all__ = ["CodeGenerator"]
_logger = logging.getLogger('utensor-cli')

class CodeGenerator(object):
  def __init__(self, model_file,
               idx_dir,
               embed_data_dir,
               trans_methods,
               output_nodes,
               save_graph=True,
               debug_cmt=True,
               **trans_kwargs):
    self.model_file = model_file
    if not os.path.exists(idx_dir):
      os.makedirs(idx_dir)
    self.idx_dir = idx_dir
    self.embed_data_dir = embed_data_dir.rstrip("/")
    self.trans_methods = trans_methods
    self.output_nodes = output_nodes
    self.save_graph = save_graph
    self.debug_cmt = debug_cmt
    self.trans_kwargs = trans_kwargs
    print ("hpplinux CodeGenerator __init__")
    print (self.__dict__)
	


  def generate(self, src_fname):
    _, ext = os.path.splitext(self.model_file)
    if ext == '.pb':
      self._generate_from_pb(src_fname)
    else:
      raise ValueError('Support only pb file')

  def _generate_from_pb(self, src_fname):
    """Generate source and header files
    """
    fname, _ = os.path.splitext(src_fname)
    graph_name, _ = os.path.splitext(os.path.basename(self.model_file))
    guard_name = fname.replace('/', '_')
    weightheader_fname = '{}_weight.hpp'.format(fname)
    header_snippet = ContextHeaderSnippet(guard_name, graph_name)
    weight_container = ContextGlobalArrayContainer()
    composer = Composer()
    header_fname = '{}.hpp'.format(fname)
    header_name = os.path.basename(header_fname)
    weightheader_name = os.path.basename(weightheader_fname)
    container = ContextSnippetsContainer(graph_name, header_name, weightheader_name)

    opFactory = OperatorFactory()

    print ("hpplinux   -----------1----")
    print ("hpplinux header_snippet : ")
    print (header_snippet.__dict__)
    print ("hpplinux container :")
    print (container.__dict__)
    graph_def = self._tf_load_graph_def(self.model_file)
    print ("hpplinux graph_def :")
    print (graph_def)
    self._expect_non_quantized(graph_def)
    ugraph = uTensorGraph(graph_def, self.output_nodes)
    print ("hpplinux ugraph : ")
    print (ugraph.__dict__)
    _logger.info("Transforming graph: %s", self.model_file)
    _logger.info("Transform pipeline: %s", ' -> '.join(self.trans_methods))
    quant_ugraph = self._transform_graph(ugraph,
                                         self.trans_methods,
                                         self.trans_kwargs)
    print ("hpplinux   -----------2----")
    print ("hpplinux quant_ugraph :")
    print (quant_ugraph.__dict__)
    _logger.info('Graph transormation done')

    if self.save_graph:
      _logger.info('Saving transformed graph')
      pkl_fname = "quant_{}.pkl".format(graph_name)
      with open(pkl_fname, 'wb') as fid:
        pickle.dump(quant_ugraph, fid)
      _logger.info('{} saved'.format(pkl_fname))

    for op_id, op_name in enumerate(quant_ugraph.topo_order):
      op_info = quant_ugraph.ops_info[op_name]
      op_type = op_info.op_type
      print (" ")
      print (" ")
      print ("hpplinux  op_name :",end='')
      print (op_name)
      print ("hpplinux  op_info :",end='')
      print (op_info)
      print ("hpplinux  op_type :",end='')
      print (op_type)
      # TODO: better abstraction for snippet
      if op_type == "Placeholder":
        parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                        op_info.op_attr)
        out_tname = op_info.output_tensors[0].name
        ref_count = parser.get('ref_counts', [0])[0]
        container.template_vars["placeholders"].append(out_tname)
        container.template_vars["ref_counts"].append(ref_count)
        header_snippet.template_vars["placeholders"].append(out_tname)
        print ("hpplinux  op_type == Placeholder")
        print ("out_tname :")
        print (out_tname)
        print ("container :")
        print (container.__dict__)
        print ("container.render() :")
        print (container.render())
        print ("header_snippet :")
        print (header_snippet.render())
      else:
        # TODO: the operator may correspond to multiple snippets (such as InlinTensor)
        # weight_container is passed to function for workaround
        print ("hpplinux  op_type != Placeholder")
        snippet = opFactory.createOperatorSnippet(op_info,
                                                  idx_dir=self.idx_dir,
                                                  embed_data_dir=self.embed_data_dir,
                                                  weight_container=weight_container)
        print ("hpplinux snippet :")
        print (snippet.__dict__)
        print ("hpplinux snippet.render :")
        print (snippet.render())
        container.add_snippet(snippet)
        print ("hpplinux container :")
        print (container.__dict__)
        print ("hpplinux container.render() :")
        print (container.render())
        print ("hpplinux weight_container :")
        print (weight_container.render())
      if self.debug_cmt:
        comments = ["<<< Operation id {}: {}".format(op_id, op_name),
                    ">>> Operation id {}: {}".format(op_id + 1, op_name)]
        cmt_snippet = CommentSnippet(comments)
        container.add_snippet(cmt_snippet)
    composer.add_snippet(container)
    print ("hpplinux composer :")
    print (composer.compose())

    if 'inline' in self.trans_methods:
      _logger.info("Generate weight file: %s", weightheader_fname)
      with open(weightheader_fname, "w") as wf:
        wf.write('// Auto generated by utensor-cli\n\n')
        wf.write(weight_container.render())
    else:
      container.remove_header('"{}"'.format(weightheader_name))
      
    _logger.info("Generate header file: %s", header_fname)
    with open(header_fname, "w") as wf:
      wf.write('// Auto generated by utensor-cli\n\n')
      wf.write(header_snippet.render())
    _logger.info("Generate source file: %s", src_fname)
    with open(src_fname, "w") as wf:
      wf.write('// Auto generated by utensor-cli\n\n')
      wf.write(composer.compose())
  
  @classmethod
  def _expect_non_quantized(cls, graph_def):
    is_quantized = False
    for node in graph_def.node:
      if node.op in ["Dequantize", "QuantizedMaxPool",
                     "QuantizeV2", "QuantizedMatMul",
                     "QuantizedRelu", "QuantizedAdd",
                     "RequantizationRange",
                     "Requantize",
                     "QuantizedReshape",
                     "QuantizedConv2D"]:
        is_quantized = True
        break
    if is_quantized:
      _logger.warning(("Expecting non-quantized graph, "
                        "graph transformation/optimization might not work properly"))

  def _transform_graph(self, ugraph, methods, trans_kwargs):
    pipeline = TransformerPipeline(methods, trans_kwargs)
    return pipeline.transform(ugraph)

  def _tf_load_graph_def(self, pb_fname):
    with tf.gfile.FastGFile(pb_fname, 'rb') as fid:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(fid.read())
    return graph_def
