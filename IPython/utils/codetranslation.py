# encoding: utf-8

"""Utilities to translate Python bytecode between different Python versions
"""

import sys
import os
import new
from functools import wraps
from collections import deque


__all__ = ['current_python_version',
           'translate_code_to_current_version',
           'translate_code']
#
# API
#

current_python_version = '.'.join(str(x) for x in sys.version_info[:2])

def translate_code_to_current_version(code, src_version):
    return translate_code(code, src_version, current_python_version)

def translate_code(code, src_version, dst_version):
    if src_version == dst_version:
        return code
    return cached_translate_code(code, src_version, dst_version)


#
# Internal utilities
#

def simple_memorize(func=None, cache=None):
    if func is None:
        return lambda func: simple_memorize(func, cache)

    if cache is None:
        cache = {}

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache
    return wrapper

def load_module(path, name=None, extra_gbls=None):
    if name is None:
        name = os.path.splitext(os.path.basename(name))[0]

    mod = new.module(name)
    gbls = vars(mod)
    gbls.update(__name__=name, __file__=path)
    gbls.update(extra_gbls or {})

    execfile(path, gbls)

    return mod


class CappedSizeDict(dict):

    def __init__(self, max_size, *args, **kwds):
        super(CappedSizeDict, self).__init__(*args, **kwds)
        self.max_size = max_size
        self.keys_order = deque()

    def __setitem__(self, key, value):
        if key in self:
            self.keys_order.remove(key)
        elif len(self) == self.max_size:
            discard_key = self.keys_order.popleft()
            del self[discard_key]

        super(CappedSizeDict, self).__setitem__(key, value)
        self.keys_order.append(key)

    def update(self, *args, **kwds):
        # hacky version of update. doesn't properly handle
        # an ordered sequence as sole argument (order matters
        # in determining what is discarded)
        for key,value in dict(*args, **kwds).iteritems():
            self[key] = value



#
# Hacks to load byteplay modules (assembler/disassembler) that target
# different Python versions
#

current_directory = os.path.dirname(os.path.abspath(__file__))

opcodes_directory = os.path.join(current_directory, 'opcodes')
assert os.path.isdir(opcodes_directory)

xbyteplay_path = os.path.join(current_directory, 'xbyteplay.py')
assert os.path.isfile(xbyteplay_path)

def load_opcodes_module(python_version):
    name = python_version.replace('.', '')
    path = os.path.join(opcodes_directory, name + '.py')
    if not os.path.exists(path):
        raise ValueError("path %s doesn't exist" % (path,))
    return load_module(path, 'opcode' + name)

xbyteplay_modules = dict()

def load_xbyteplay(python_version):
    name = 'xbyteplay' + python_version.replace('.', '')
    opcode = load_opcodes_module(python_version)
    mod = load_module(xbyteplay_path, name,
                      dict(python_version=python_version,
                           opcode=opcode))
    xbyteplay_modules[name] = mod
    xbyteplay_modules[python_version] = mod
    return mod

@simple_memorize
def get_xbyteplay(python_version):
    return load_xbyteplay(python_version)

def decompile_code(code, python_version):
    return get_xbyteplay(python_version).Code.from_code(code)

def compile_code(asm, python_version):
    if asm.python_version != python_version:
        raise ValueError("byteplay.Code isn't of expected version")
    return asm.to_code()


#
# Translation
#
# Define a collection of functions that can translate between specific
# Python versions of bytecode (e.g. 2.6 -> 2.7). Next a directed graph
# is constructed with the arcs being these specific translations. Then
# the translation between any two arbitrary Python versions is accomplished
# by finding the shortest path between these two versions.
#


default_translate_cache_size = 10

translate_cache = CappedSizeDict(default_translate_cache_size)

@simple_memorize(cache=translate_cache)
def cached_translate_code(code, src_version, dst_version):
    src_asm = decompile_code(code, src_version)
    dst_asm = translate_assembly(src_asm, src_version, dst_version)
    return compile_code(dst_asm, dst_version)

def translate_assembly(src_asm, src_version, dst_version):
    asm = src_asm
    for translation in get_translation_path(src_version, dst_version):
        asm = translation.translate(asm)
    return asm

@simple_memorize
def get_translation_path(src_version, dst_version):
    return list(translation_graph.find_optimal_path(src_version, dst_version).iter_arcs())


#
# Graph utilities
#

class Node(object):

    def __init__(self, value):
        self.value = value
        self.out_arcs = []
        self.in_arcs = []

    def find_paths(self, dst):
        for path in self.iter_all_out_paths():
            if path.get_node(-1) == dst:
                yield path

    def iter_all_out_paths(self, current_path=None):
        if current_path is None:
            current_path = Path()
        elif self in current_path:
            return

        for out_arc in self.out_arcs:
            path = current_path.extend(out_arc)
            yield path

            for sub_path in out_arc.dst_node.iter_all_out_paths(path):
                yield sub_path

class Arc(object):

    def __init__(self, src_node, dst_node):
        self.src_node = src_node
        self.dst_node = dst_node


class Path(object):

    def __init__(self, arcs=()):
        self.arcs = arcs

    def extend(self, arc):
        assert isinstance(arc, Arc)
        return self.__class__(self.arcs + (arc,))

    def __contains__(self, op):
        if isinstance(op, Node):
            return op in self.iter_nodes()
        elif isinstance(op, Arc):
            return op in self.iter_arcs()
        else:
            raise TypeError

    def iter_nodes(self):
        if self.arcs:
            yield self.arcs[0].src_node
        for arc in self.arcs:
            yield arc.dst_node

    def iter_arcs(self):
        return iter(self.arcs)

    def __nonzero__(self):
        return bool(self.arcs)

    @property
    def n_arcs(self):
        return len(self.arcs)

    @property
    def n_nodes(self):
        return 1+len(self.arcs) if self.arcs else 0

    def get_arc(self, index):
        return self.arcs[index]

    def get_node(self, index):
        if index < 0:
            index = index + self.n_nodes
        if index==0:
            return self.arcs[0].src_node
        return self.arcs[index-1].dst_node


class DirectedGraph(object):

    node_factory = Node
    arc_factory = Arc

    def __init__(self):
        self.nodes = {}

    def get_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = self.node_factory(value)
        return self.nodes[value]

    def add_arc(self, src_value, dst_value, *extra_args, **extra_kwds):
        src_node = self.get_node(src_value)
        dst_node = self.get_node(dst_value)
        arc =  self.arc_factory(src_node, dst_node, *extra_args, **extra_kwds)
        src_node.out_arcs.append(arc)
        dst_node.in_arcs.append(arc)
        return arc

    def find_optimal_path(self, src_value, dst_value, metric=lambda path: path.n_arcs, optimal=min):
        paths = self.find_paths(src_value, dst_value)
        return optimal(paths, key=metric)

    def find_paths(self, src_value, dst_value):
        return self.get_node(src_value).find_paths(self.get_node(dst_value))

#
# Translations
#

class Translation(Arc):

    def __init__(self, src_node, dst_node, translator_class):
        super(Translation, self).__init__(src_node, dst_node)
        self.src_version = src_node.value
        self.dst_version = dst_node.value
        self.translator_class = translator_class

    def translate(self, src_asm):
        validate_asm(src_asm)
        assert src_asm.python_version == self.src_version
        dst_asm = self.translator_class(src_asm).translate()
        assert dst_asm.python_version == self.dst_version
        validate_asm(dst_asm)
        return dst_asm


class TranslationGraph(DirectedGraph):

    arc_factory = Translation

    def add_translation(self, translator_class):
        return self.add_arc(translator_class.src_version,
                            translator_class.dst_version,
                            translator_class=translator_class)


class TranslationError(Exception):

    def __init__(self, src_version, dst_version, msg):
        self.src_version = src_version
        self.dst_version = dst_version
        self.msg = msg

    def __str__(self):
        return 'translation %s -> %s failed: %s' % (self.src_version, self.dst_version, self.msg)


def copy_asm(asm, version=None, **kwds):
    xbyteplay = get_xbyteplay(version or asm.python_version)
    init_args = 'code freevars args varargs varkwargs newlocals name filename firstlineno docstring'.split()
    for arg in init_args:
        if arg not in kwds:
            kwds[arg] = getattr(asm, arg)
    kwds['code'] = xbyteplay.CodeList(kwds['code'])
    args = [kwds[arg] for arg in init_args]
    return xbyteplay.Code(*args)

def asm_rebuild_helper(asm, func):
    return copy_asm(asm, code=func(asm.code))

def get_op_version(op):
    return xbyteplay_modules[op.__class__.__module__].python_version

def validate_asm(asm):
    if any(get_op_version(op) != asm.python_version for op,arg in asm.code):
        raise ValueError("assembly contains operations from other python bytecode versions")


class BaseTranslator(object):

    def __init__(self, src_asm):
        self.src_asm = src_asm
        assert src_asm.python_version == self.src_version

    def translation_error(self, msg, *args):
        if args:
            msg %= args
        raise TranslationError(self.src_version, self.dst_version, msg)

    def translate_operations_by_name(self, asm):
        '''Handle opcodes that have the same name in each opcode version
           (e.g. DUP_TOP). Can be dangerous if they have the same name, but
           different semantics in each version (e.g. YIELD_VALUE in 2.4 vs 2.5).
           This procedure should be ran last to deal with the simple cases of
           operations invariant between the two Python versions.
        '''
        operations_map = self.get_operations_name_map()
        dst_xbyteplay = get_xbyteplay(self.dst_version)
        src_xbyteplay = get_xbyteplay(self.src_version)

        labels_map = {}
        def get_label(src_label):
            if src_label not in labels_map:
                labels_map[src_label] = dst_xbyteplay.Label()
            return labels_map[src_label]

        acc_code = []
        for src_op,arg in asm.code:
            if get_op_version(src_op) == self.dst_version:
                dst_op = src_op
            elif src_op is src_xbyteplay.SetLineno:
                dst_op = dst_xbyteplay.SetLineno
            elif isinstance(src_op, src_xbyteplay.Label):
                dst_op = get_label(src_op)
            elif src_op not in operations_map:
                self.translation_error("no named translation for %s", src_op)
            else:
                dst_op = operations_map[src_op]

            if src_op in dst_xbyteplay.hasjump:
                arg = get_label(arg)

            acc_code.append([dst_op, arg])

        return copy_asm(asm, version=self.dst_version, code=acc_code)

    @classmethod
    @simple_memorize
    def get_operations_name_map(cls):
        src_xbyteplay = get_xbyteplay(cls.src_version)
        dst_xbyteplay = get_xbyteplay(cls.dst_version)
        return dict((src_op, dst_xbyteplay.opmap[name])
                    for name,src_op in src_xbyteplay.opmap.iteritems()
                    if name in dst_xbyteplay.opmap)


#
# 2.4 -> 2.5
#

### TODO

#
# 2.5 -> 2.6
#

### TODO

#
# 2.6 -> 2.7
#

class Translate26to27(BaseTranslator):

    src_version = '2.6'
    dst_version = '2.7'

    def translate(self):
        asm = self.src_asm
        asm = self.upgrade_conditions(asm)
        asm = self.fix_list_append(asm)
        asm = self.translate_operations_by_name(asm)
        return asm

    def upgrade_conditions(self, asm):
        # Replace all JUMP_IF_FALSE/TRUE with 
        # DUP_TOP POP_JUMP_IF_FALSE/TRUE. There are definitely
        # more efficient ways to use these new instructions, but
        # likely not worth the effort to implement these more
        # complex transformations just yet.

        def rebuild_ops(ops):
            src_xbyteplay = get_xbyteplay(self.src_version)
            dst_xbyteplay = get_xbyteplay(self.dst_version)

            for op,arg in ops:
                if op == src_xbyteplay.JUMP_IF_FALSE:
                    yield dst_xbyteplay.DUP_TOP, None
                    yield dst_xbyteplay.POP_JUMP_IF_FALSE, arg
                elif op == src_xbyteplay.JUMP_IF_TRUE:
                    yield dst_xbyteplay.DUP_TOP, None
                    yield dst_xbyteplay.POP_JUMP_IF_TRUE, arg
                else:
                    yield op, arg

        return asm_rebuild_helper(asm, rebuild_ops)

    def fix_list_append(self, asm):
        # In 2.6 LIST_APPEND pops the list from the stack, where as
        # in 2.7 the list remains on the stack. We simply add a POP_TOP
        # to duplicate 2.6 behavior.

        def rebuild_ops(ops):
            xbyteplay = get_xbyteplay(self.src_version)

            for op,arg in ops:
                if op == xbyteplay.LIST_APPEND:
                    yield op, 1
                    yield xbyteplay.POP_TOP, None
                else:
                    yield op, arg

        return asm_rebuild_helper(asm, rebuild_ops)


#
# Registration of translators
#

translation_graph = TranslationGraph()
translation_graph.add_translation(Translate26to27)
