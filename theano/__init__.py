"""
Theano is an optimizing compiler in Python, built to evaluate
complicated expressions (especially matrix-valued ones) as quickly as
possible.  Theano compiles expression graphs (see :doc:`graph` ) that
are built by Python code. The expressions in these graphs are called
`Apply` nodes and the variables in these graphs are called `Variable`
nodes.

You compile a graph by calling `function`, which takes a graph, and
returns a callable object.  One of theano's most important features is
that `function` can transform your graph before compiling it.  It can
replace simple expressions with faster or more numerically stable
implementations.

To learn more, check out:

- Op List (:doc:`oplist`)

The markup language used in the docstrings is ReStructured Text,
which may be rendered with Sphinx. A rendered version is
maintained at http://www.deeplearning.net/software/theano/library/

"""

__docformat__ = "restructuredtext en"

# Set a default logger. It is important to do this before importing some other
# theano code, since this code may want to log some messages.
import logging

theano_logger = logging.getLogger("theano")

# The user might not want the default logger, but we cannot know
# without looking at config; Looking at config could cause logging.
# We know the default handler would write to stderr, so we create a
# temporary handler to write to a buffer. If the default handler is required, we
# copy the content of the buffer to stderr and pretend nothing happened.
import cStringIO as sio
from sys import stderr as err
temp_stream = sio.StringIO()
logging_temp_handler = logging.StreamHandler(temp_stream)

logging_default_formatter = logging.Formatter(
        fmt='%(levelname)s (%(name)s): %(message)s')
logging_temp_handler.setFormatter(logging_default_formatter)
theano_logger.addHandler(logging_temp_handler)
theano_logger.setLevel(logging.WARNING)

try:
    from theano.configdefaults import config
except:
    # An exception occurred, somehow.
    # Rather than taking chances, we will dump the buffer
    # to stderr. At this point, the user has other preoccupations
    # than duplicated logs.
    err.write(temp_stream.getvalue())
    raise

theano_logger.removeHandler(logging_temp_handler)
del logging_temp_handler

if config.handle_log:
    err.write(temp_stream.getvalue())
    logging_default_handler = logging.StreamHandler()
    logging_default_handler.setFormatter(logging_default_formatter)
    theano_logger.addHandler(logging_default_handler)

del temp_stream, err, sio


# Version information.
from theano.version import version as __version__

from theano.gof import \
    CLinker, OpWiseCLinker, DualLinker, Linker, LocalLinker, PerformLinker, \
    Container, \
    InconsistencyError, FunctionGraph, \
    Apply, Variable, Constant, \
    Op, OpenMPOp, \
    opt, \
    toolbox, \
    Type, Generic, generic, \
    object2, utils

from theano.compile import \
    SymbolicInput, In, \
    SymbolicOutput, Out, \
    Mode, \
    predefined_modes, predefined_linkers, predefined_optimizers, \
    FunctionMaker, function, function_dump, OpFromGraph, \
    ProfileMode, ProfileStats, \
    Param, shared, as_op

from theano.misc.safe_asarray import _asarray

from theano.printing import pprint, pp

from theano.scan_module import scan, map, reduce, foldl, foldr, clone

from theano.updates import OrderedUpdates

# scan_module import above initializes tensor and scalar making these imports redundant
#import tensor
#import scalar
#we don't import by default as we don't want to force having scipy installed.
#import sparse

from theano.gradient import Rop, Lop, grad, subgraph_grad

# This need to be before the init of GPU, as it add config variable
# needed during that phase.
import theano.tests
if hasattr(theano.tests, "TheanoNoseTester"):
    test = theano.tests.TheanoNoseTester().test
else:
    def test():
        raise ImportError("The nose module is not installed."
                          " It is needed for Theano tests.")

if config.device.startswith('gpu') or config.init_gpu_device.startswith('gpu'):
    import theano.sandbox.cuda
    # We can't test the driver during import of theano.sandbox.cuda as
    # this cause circular import dependency. So we also test it manually
    # after the import
    if theano.sandbox.cuda.cuda_available:
        import theano.sandbox.cuda.tests.test_driver

        theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()

if config.device.startswith('cuda') or config.device.startswith('opencl') or \
        config.gpuarray.init_device != '':
    import theano.sandbox.gpuarray

# Use config.numpy to call numpy.seterr
import numpy

if config.numpy.seterr_all == 'None':
    _all = None
else:
    _all = config.numpy.seterr_all
if config.numpy.seterr_divide == 'None':
    _divide = None
else:
    _divide = config.numpy.seterr_divide
if config.numpy.seterr_over == 'None':
    _over = None
else:
    _over = config.numpy.seterr_over
if config.numpy.seterr_under == 'None':
    _under = None
else:
    _under = config.numpy.seterr_under
if config.numpy.seterr_invalid == 'None':
    _invalid = None
else:
    _invalid = config.numpy.seterr_invalid
numpy.seterr(
    all=_all,
    divide=_divide,
    over=_over,
    under=_under,
    invalid=_invalid)
del _all, _divide, _over, _under, _invalid

## import scalar_opt

### This is defined here because it is designed to work across symbolic
#   datatypes (Sparse and Tensor)


def dot(l, r):
    """Return a symbolic matrix/dot product between l and r """
    rval = NotImplemented
    e0, e1 = None, None

    if rval == NotImplemented and hasattr(l, '__dot__'):
        try:
            rval = l.__dot__(r)
        except Exception, e0:
            rval = NotImplemented
    if rval == NotImplemented and hasattr(r, '__rdot__'):
        try:
            rval = r.__rdot__(l)
        except Exception, e1:
            rval = NotImplemented
    if rval == NotImplemented:
        raise NotImplementedError("Dot failed for the following reasons:",
                                  (e0, e1))
    return rval


def get_scalar_constant_value(v):
    """return the constant scalar(0-D) value underlying variable `v`

    If v is the output of dimshuffles, fills, allocs, rebroadcasts, cast
    this function digs through them.

    If theano.sparse is also there, we will look over CSM op.

    If `v` is not some view of constant data, then raise a
    tensor.basic.NotScalarConstantError.
    """
    # Is it necessary to test for presence of theano.sparse at runtime?
    if 'sparse' in globals() and isinstance(v.type, sparse.SparseType):
        if v.owner is not None and isinstance(v.owner.op, sparse.CSM):
            data = v.owner.inputs[0]
            return tensor.get_scalar_constant_value(data)
    return tensor.get_scalar_constant_value(v)


def sparse_grad(var):
    """This function return a new variable whose gradient will be
    stored in a sparse format instead of dense.

    Currently only variable created by AdvancedSubtensor1 is supported.
    i.e. a_tensor_var[an_int_vector].

    .. versionadded:: 0.6rc4
    """
    assert isinstance(var.owner.op, tensor.AdvancedSubtensor1)
    ret = var.owner.op.__class__(sparse_grad=True)(*var.owner.inputs)
    return ret


__import__('theano.tensor.shared_randomstreams')
