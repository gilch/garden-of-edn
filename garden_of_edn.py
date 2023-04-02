# Copyright 2022, 2023 Matthew Egan Odendahl
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""A Garden of EDN parsers. Bringing the Python to EDN.

These parsers are not validators. Behavior when given invalid EDN is
undefined. EDN is not especially well-specified to begin with.

These parsers are not serializers. They read EDN and render it as Python
objects; they don't serialize Python objects into EDN format.
"""
import ast
import builtins
import doctest
import re
from abc import ABCMeta, abstractmethod
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, reduce
from importlib import import_module
from itertools import takewhile
from operator import methodcaller
from typing import Iterator
from unittest.mock import sentinel
from uuid import UUID

import hissp
from hissp.compiler import MACROS, Compiler
from hissp.munger import munge
from pyrsistent import plist, pmap, pset, pvector

TOKENS = re.compile(
    r"""(?x)
    (?P<_comment>;.*)
    |(?P<_whitespace>[,\s]+)
    |(?P<_rcub>})
    |(?P<_rpar>\))
    |(?P<_rsqb>])
    |(?P<_discard>\#_)
    |(?P<_set>\#{)
    |(?P<_map>{)
    |(?P<_list>\()
    |(?P<_vector>\[)
    |(?P<_tag>
      \#[^\W\d_]  # Tags must start with # followed by an alphabetic.
      [-+.:#*!?$%&=<>/\w]*  # Any number of any symbol characters.
     )
    |(?P<_string>
      "  # Open quote.
        (?:[^"\\]  # Any non-magic character.
           |\\[trn\\"]  # Backslash only if paired, including with newline.
        )*  # Zero or more times.
      "  # Close quote.
     )
    |(?P<_atom>[^]}),\s]+|\\.)
    |(?P<_error>.)
    """
)
ATOMS = re.compile(
    r"""(?x)
    (?P<_int>[-+]?(?:\d|[1-9]\d+)N?)
    |(?P<_float>
      [-+]?(?:\d|[1-9]\d+)
      (?:.\d+)?
      (?:[eE][-+]?\d+)?
      M?
    )
    |(?P<keyword> # Unclear from spec, but assume no empty EDN Keywords.
      :
      [-+.#*!?$%&=<>/\w] # Second character cannot be :.
      [-+.:#*!?$%&=<>/\w]*
     )
    |(?P<_symbol>
      [-+.]
      |(?:(?:[*!?$%&=<>/]|[^\W\d]) # Always valid at start. Not [:#\d].
          |[-+.](?:[-+.:#*!?$%&=<>/]|[^\W\d]) # [-+.] can't be followed by a \d
       )[-+.:#*!?$%&=<>/\w]*)
    |(?P<_char>\\(?:newline|return|space|tab|u[\dA-Fa-f]{4}|\S))
    |(?P<_error>.)
    """
)
SINGLES = re.compile(r"""(?P<nil>nil)|(?P<bool>true|false)|(?P<symbol>.+)""")

def _kv(m):
    return m.lastgroup, m.group()

def tokenize(edn):
    for m in TOKENS.finditer(edn):
        k, v = _kv(m)
        if k=='_atom':
            k, v = _kv(ATOMS.fullmatch(v))
        if k=='_symbol':
            k, v = _kv(SINGLES.fullmatch(v)) or ('symbol', v)
        if k in {'_comment','_whitespace'}:
            continue
        if k=='_error':
            raise ValueError(m.pos)
        yield k, v

class AbstractEDN(metaclass=ABCMeta):
    """AbstractEDN is a highly customizable EDN parser base class.

    Takes an EDN string and optionally a mapping of tag names
    (without the leading #) to tag rendering functions. These are used
    to update a dict containing rendering functions for #inst and
    #uuid, the EDN builtin tags (which can be overriden). Tag
    renderers must accept the next Python object parsed from the EDN
    and render a Python object appropriate for the tag. The tag
    method is used as a fallback when no tag rendering function can
    be found. By default, it raises a KeyError for the tag name.

    Each of the built-in EDN atom types has a corresponding abstract
    method: symbol, string, keyword, bool, nil, float, floatM, int,
    intN and char. They must accept the token string and render a
    Python object. The "N" and "M" are removed. Char tokens are
    preprocessed per the spec.

    Each EDN collection type has a corresponding abstract method.

    The set and map methods must accept a tuple of parsed elements
    and are expected to render a suitable collection of them. There
    are also cset and cmap fallbacks in case of unhashable types in a
    set or map (usually the result of composite keys, hence the
    names) which must have raised a TypeError in the map or set
    methods. The default implementations of cset and cmap fall back
    to using the vector method. The tuple passed to map (and cmap)
    contains key-value pairs (as is most natural for Python), rather
    than alternating key and value elements as written in EDN maps.

    Parsed objects meant for map keys or set elements are passed to the
    key method, whose return value is used instead. By default, key
    returns the object unchanged, but overrides may replace an
    unhashable object with a hashable one, if desired.

    The list and vector methods must accept an iterator of the parsed
    elements and are expected to render a suitable collection of them.
    """
    def read(self):
        return self._parse()
    def __init__(self, edn, tags=(), **kwargs):
        self.tokens = tokenize(edn)
        self.tags = dict(uuid=UUID, inst=datetime.fromisoformat)
        self.tags.update(tags)
    def _tokens_until(self, k):
        return takewhile(lambda kv: kv[0] != k, self.tokens)
    def _parse_until(self, k):
        return self._parse(self._tokens_until(k))
    def _discard(self, v):
        next(self._parse())
    def _set(self, v):
        elements = tuple(self.key(k) for k in self._parse_until('_rcub'))
        try:
            return self.set(elements)
        except TypeError:
            return self.cset(elements)
    def _map(self, v):
        ikvs = self._parse_until('_rcub')
        pairs = tuple((self.key(k),next(ikvs)) for k in ikvs)
        try:
            return self.map(pairs)
        except TypeError:
            return self.cmap(pairs)
    def _list(self, v):
        return self.list(self._parse_until('_rpar'))
    def _vector(self, v):
        return self.vector(self._parse_until('_rsqb'))
    def _tag(self, v: str):
        return self.tags.get(v[1:], partial(self.tag, v[1:]))(next(self._parse()))
    def _string(self, v):
        return self.string(ast.literal_eval(v.replace('\n',R'\n')))
    def _float(self, v: str):
        return self.floatM(v[:-1]) if v.endswith('M') else self.float(v)
    def _int(self, v: str):
        return self.intN(v[:-1]) if v.endswith('N') else self.int(v)
    def _char(self, v):
        v = v[1:]
        v = {'newline':'\n','return':'\r','space':' ','tab':'\t'}.get(v,v)
        if v.startswith('u'):
            v = ast.literal_eval(Rf"'\{v}'")
        return self.char(v)
    def _parse(self, tokens=None):
        for k, v in tokens or self.tokens:
            y = getattr(self, k)(v)
            if k!='_discard':
                yield y
    # The remainder are meant for overrides.
    def key(self, k): return k
    def tag(self, tag, element): raise KeyError(tag)
    def cset(self, elements: tuple): return self.vector(elements)
    def cmap(self, elements: tuple): return self.vector(elements)
    @abstractmethod
    def list(self, elements: Iterator): ...
    @abstractmethod
    def vector(self, elements: Iterator): ...
    @abstractmethod
    def set(self, elements: tuple): ...
    @abstractmethod
    def map(self, elements: tuple): ...
    @abstractmethod
    def symbol(self, v: str): ...
    @abstractmethod
    def string(self, v: str): ...
    @abstractmethod
    def keyword(self, v: str): ...
    @abstractmethod
    def bool(self, v: str): ...
    @abstractmethod
    def nil(self, v: str): ...
    @abstractmethod
    def float(self, v: str): ...
    @abstractmethod
    def floatM(self, v: str): ...
    @abstractmethod
    def int(self, v: str): ...
    @abstractmethod
    def intN(self, v: str): ...
    @abstractmethod
    def char(self, v: str): ...

class BuiltinEDN(AbstractEDN):
    R"""Simple EDN parser.

    Renders each EDN type as the most natural equivalent builtin type,
    making the resulting data easy to use from Python.

    The 20% solution for 80% of use cases. Does not implement the full
    EDN spec, but should have no trouble parsing a typical .edn config
    file.
    >>> [*BuiltinEDN(R'42 4.2 true nil').read()]
    [42, 4.2, True, None]

    However, this means it throws away information and can't round-trip
    back to the same EDN. Keywords, strings, symbols, and characters all
    become strings, because idiomatic Python uses the str type for all
    of these use cases. (ClojureScript will also use strings for chars.)
    >>> [*BuiltinEDN(R'"foo" :foo foo \x').read()]
    ['foo', ':foo', 'foo', 'x']

    If this is a problem for your use case, you can override one of the
    methods to return a different type.

    Mixing numeric type keys or set elements is inadvisable per the EDN
    spec, so this is rarely an issue in practice, but Python's equality
    semantics differ from EDN's: numeric types are equal when they
    have the same value (and bools are treated as 1-bit ints),
    regardless of type and precision. ClojureScript has a similar
    problem, treating all numbers as floats.
    >>> next(BuiltinEDN(R'{false 1, 0 2, 0N 3, 0.0 4, 0M 5}').read())
    {False: 5}

    Collections simply use the Python collection with the same brackets.
    >>> [*BuiltinEDN(R'#{1}{2 3}(4)[5]').read()]
    [{1}, {2: 3}, (4,), [5]]

    EDN's collections are immutable, and are valid elements in EDN's set
    and as EDN's map keys, however, Python's builtin mutable collections
    (set, dict, and list) are unhashable, and therefore invalid in sets
    and as keys. In practice, most EDN data doesn't do this; keys are
    nearly always keywords or strings, but in those rare cases, this
    parser will fall back to using lists rather than dicts or sets.

    >>> next(BuiltinEDN(R'{[1] 1, [2] 2}').read())
    [([1], 1), ([2], 2)]
    >>> next(BuiltinEDN(R'#{#{}}').read())
    [set()]
    """
    set = set
    map = dict
    list = tuple
    vector = builtins.list
    char = string = str
    intN = int = int
    floatM = float = float
    keyword = symbol = str
    nil = bool = {'false':False, 'true':True}.get

class StandardEDN(BuiltinEDN):
    """Handles more cases, using only standard-library types.

    But at the cost of being a little harder to use than BuiltinEDN.
    More imports are used and some types are used in unnatural ways.

    Using only standard library types means pickled results can be
    unpickled in an environment without garden_of_edn installed.

    EDN set and vector types now map to frozenset and tuple,
    respectively, which are hashable as long as their elements are,
    allowing them to be used as keys and in sets.
    >>> next(StandardEDN(R'{[1] 1, (2) 2}').read())
    {(1,): 1, (2,): 2}
    >>> next(StandardEDN('#{#{}}').read())
    frozenset({frozenset()})

    This means that vectors and lists are no longer distinguishable, but
    (in practice) this distinction usually doesn't matter. Having two
    sequence type literals is somewhat useful in Clojure, but redundant
    in EDN. They compare equal when used in maps or sets anyway.
    If this matters for your use case, you can override one of them.

    EDN map types still render to dict. No hashable mapping type is
    available in the standard library. While the
    `types.MappingProxyType` is an immutable view, it is still not
    hashable because the underlying mapping may still be mutable.

    List is still used as a fallback for unhashable elements.
    There's not much point in using a tuple here, since a tuple with
    an unhashable element is itself unhashable.
    >>> next(StandardEDN(R'{{1 1} 2, {2 2} 4}').read())
    [({1: 1}, 2), ({2: 2}, 4)]
    >>> next(StandardEDN(R'#{{1 1} 2 {2 2} 4}').read())
    [{1: 1}, 2, {2: 2}, 4]

    Symbol and keyword types map to `unittest.mock.sentinel`,
    a standard-library type meant for unit testing, but with the
    interning semantics desired for keywords: the same keyword always
    produces the same object. Using the same type for both is allowed
    by the spec, because they remain distinguishable by the leading
    colon.
    >>> next(StandardEDN(':foo').read())
    sentinel.:foo
    >>> _ is next(StandardEDN(':foo').read())
    True

    Python has a perfectly good bool type, but because EDN equality
    is different, it would cause collections to fail to round-trip
    in some cases.

    True is a special case of 1 and False 0 in Python, so the first
    values were overwritten.
    >>> next(BuiltinEDN('{0 0, 1 1, false 2, true 3}').read())
    {0: 2, 1: 3}

    EDN doesn't consider these keys equal, so data was lost.
    StandardEDN can handle this without loss, by using the same
    sentinel type for true as well. sentinel.false could also be used
    without abiguity, but b'' has the advantage of being falsy, while
    still never comparing equal to any standard EDN type.
    >>> next(StandardEDN('{0 0, 1 1, false 2, true 3}').read())
    {0: 0, 1: 1, b'': 2, sentinel.true: 3}

    The precision types are now distinguishable. A denominator-1
    fraction is a bit less natural. Python 2 used to have a separate
    int and long type, but now its int is arbitrary-precision and it
    lacks a fixed-precision type. Because int is expected to be more
    common than intN, it gets the builtin, and intN gets something
    else.
    >>> next(StandardEDN('[0 0N 0M 0.0]').read())
    (0, Fraction(0, 1), Decimal('0'), 0.0)

    However, any numeric type of the same value still compares equal.
    Mixing numeric types in a hash table is inadvisable, per the EDN
    spec, and ClojureScript has similar problems, so this is rarely
    an issue in practice.
    >>> next(StandardEDN('#{0 0N 0M 0.0}').read())
    frozenset({0})
    """
    set = frozenset
    cmap = cset = list
    vector = tuple
    floatM = Decimal
    # The above thee are sensible choices, although Decimal is not in
    # builtins. The remainder are unnatural, but work.
    intN = Fraction  # Denominator 1.
    keyword = symbol = partial(getattr, sentinel)  # Meant for tests only.
    nil = bool = {'false':b'', 'true':sentinel.true}.get
    tag = methodcaller  # Defers a call, but won't actually be a method.

class HashBox:
    """Wrapper to make keys behave like EDN.

    Python types have two issues representing EDN keys.

    First, EDN maps use different equality semantics. In Python,
    numbers are equal if their values are equal, even if they have
    different types (and bools are also numbers). In EDN, numbers
    must be of the same type to be equal. A HashBox is equal to an
    object only if the object is also of type HashBox, and their key
    objects are of exactly the same type and are equal.
    >>> HashBox(1) == HashBox(1)
    True
    >>> 1 == 1.0 # int and float can compare equal
    True
    >>> HashBox(1) == HashBox(1.0)  # But not when boxed.
    False

    (This is typically not an issue for numbers in practice as mixing
    numeric types like this is not recommended in EDN. ClojureScript
    also has trouble with this. Booleans are a different story.)

    Second, for most EDN collection types, the most natural Python
    analogue is unhashable. The result of custom tags may likewise be
    unhashable. If its key is unhashable, HashBox will fall back to
    using the hash of its type, a characteristic assumed to be
    immutable.
    >>> hash(HashBox({})) == hash(dict)
    True

    If a hash table contains many HashBox keys, it may suffer
    degraded performance as keys cannot be dispersed over as many
    buckets when they produce equal hashes. This is likely no worse
    than the alternative of scanning through a list for a key,
    as they are at least dispersed by type, and any hashable boxed
    keys are also dispersed by value.

    Also, mutating anything used as a hash table key is a bad idea,
    liable to cause surprises. HashBox enables this and can do nothing
    to prevent it. Use with care.
    """
    def __init__(self, k):
        self.k = k
    def __eq__(self, other):
        return (type(self) is type(other)
                and (type(self.k), self.k) == (type(other.k), other.k))
    def __hash__(self):
        try:
            return hash(self.k)
        except TypeError:
            return hash(type(self.k))
    def __repr__(self):
        return f'{type(self).__name__}({self.k!r})'

class BoxedEDN(StandardEDN):
    """Uses HashBox for keys.

    Unlike the simpler parsers, there are no cases expected to lose
    data. This a round-tripping parser.

    >>> next(BoxedEDN(R'{{1 1} 2, {2 2} 4}').read())
    {HashBox({HashBox(1): 1}): 2, HashBox({HashBox(2): 2}): 4}
    >>> next(BoxedEDN(R'#{{1 1} 2 {2 2} 4}').read())
    frozenset({HashBox(2), HashBox({HashBox(1): 1}), HashBox(4), HashBox({HashBox(2): 2})})
    >>> next(BoxedEDN('#{0 0N 0M 0.0 false}').read())
    frozenset({HashBox(0), HashBox(Fraction(0, 1)), HashBox(False), HashBox(Decimal('0')), HashBox(0.0)})

    Due to the use of a non-standard type, unlike StandardEDN,
    unpickling the results in another environment may require a
    Garden of EDN install there.

    Bools use BuiltinEDN's method. They will get wrapped anyway,
    so there's no reason not to use the more natural types. Rendered
    types are otherwise as StandardEDN.
    """
    bool = BuiltinEDN.bool
    key = HashBox

class LilithHissp(BuiltinEDN):
    """Parses to Hissp. Allows Python programs to be written in EDN.

    The compiled output doesn't necessitate any installation to run.
    """
    def compile(self):
        """Yields the Python compilation of each Hissp form.

        EDN is interpreted as Hissp. Reading uses the provided tags and
        compilation uses the provided ns.
        Compiles the forms without executing them.
        Forms that have not been executed in the ns cannot affect
        the compilation of subsequent forms.
        """
        for x in self.read(): yield self.compiler.compile([x])
    def exec(self) -> str:
        """Compiles and executes each Hissp form.

        Returns the compiled Python.
        """
        try:
            self.compiler.evaluate = True
            return self.compiler.compile(self.read())
        finally:
            self.compiler.evaluate = False
    def __init__(self, edn, tags=(), *, qualname='__main__', ns=None, **kwargs):
        self.compiler = Compiler(qualname=qualname, ns=ns, evaluate=False)
        super().__init__(edn, tags, **kwargs)
    def string(self, v):
        return f'({repr(v)})'
    keyword = str
    floatM = Decimal
    def symbol(self, v):
        if v != '/':
            v = v.replace('/', '..')
        if v == '.':
            return ':'
        return munge(v)
    def tag(self, tag, element):
        extras = ()
        if tag == 'hissp/.':  # inject
            return eval(hissp.readerless(element, self.compiler.ns), self.compiler.ns)
        if tag == 'hissp/!':  # extra
            tag, *extras, element = element
        if tag == 'hissp/$':  # munge
            return munge(ast.literal_eval(element))
        *module, function = tag.split("/")
        if not module or re.match(rf"{MACROS}\.[^.]+$", function):
            function += munge("#")
        module = import_module(*module) if module else self.compiler.ns[MACROS]
        f = reduce(getattr, function.split("."), module)
        args, kwargs = hissp.reader._parse_extras(extras)
        with self.compiler.macro_context():
            return f(element, *args, **kwargs)

class PyrMixin:
    """Mixin to make an EDN parser use Pyrsistent data structures.

    These fit EDN much better that Python's builtin collection types.
    """
    set = staticmethod(pset)
    map = staticmethod(pmap)
    list = staticmethod(plist)
    vector = pvector  # nondescriptor

class PyrBuiltinEDN(PyrMixin, BuiltinEDN):
    """Adds Pyrsistent collections to BuiltinEDN.

    Unpickling the results in another environment requires
    Pyrsistent, but not Garden of EDN.
    """

class PyrStandardEDN(PyrMixin, StandardEDN):
    """Adds Pyrsistent collections to StandardEDN.

    Unpickling the results in another environment requires
    Pyrsistent, but not Garden of EDN.
    """

class PyrBoxedEDN(PyrMixin, BoxedEDN):
    """Adds Pyrsistent collections to BoxedEDN

    Pyrsistent collections are already hashable (if their elements
    are), but the equality problem remains and tags may still generate
    unhashable keys, so boxed keys are still required for a round-trip
    guarantee.

    Unpickling the results in another environment requires
    Pyrsistent and Garden of EDN.
    """

class PandoraHissp(PyrMixin, LilithHissp):
    """Adds Pyrsistent collections to LilithHissp, except lists.

    The compiled output is expected to require Pyrsistent to run.
    """
    list = tuple

if __name__ == '__main__':
    doctest.testmod()

# TODO: moar doctests
# FIX: ns coalesce bug in readerless
# TODO: make _parse_extras public
# TODO: import munge in hissp.
# TODO: import Compiler in hissp.
# TODO: HisspEDN repl?
# TODO: basic pretty printer
# TODO: serializers?
