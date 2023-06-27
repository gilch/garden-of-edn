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
import pathlib
import re
from abc import ABCMeta, abstractmethod
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, reduce
from importlib import import_module
from importlib.abc import FileLoader
from importlib.util import spec_from_loader
from itertools import takewhile
from operator import methodcaller
from typing import Iterator
from unittest.mock import sentinel
from uuid import UUID

import hissp
from hissp.compiler import MACROS
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
    |(?P<_atom>[^]}),;\s]+|\\.)
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

def tokenize(edn: str, filename=None):
    for m in TOKENS.finditer(edn):
        try:
            k, v = _kv(m)
            if k=='_atom': k, v = _kv(ATOMS.fullmatch(v))
            if k=='_symbol': k, v = _kv(SINGLES.fullmatch(v)) or ('symbol', v)
            if k in {'_comment','_whitespace'}: continue
            if k=='_error': raise ValueError
            yield k, v
        except Exception as e:
            lineno = edn.count('\n', 0, m.start())
            offset = m.start() - edn.rfind('\n', 0, m.start())
            details = filename, lineno+1, offset, edn.split('\n')[lineno]
            raise SyntaxError("Couldn't tokenize EDN", details) from e

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
    def __init__(self, edn: str, tags=(), filename=None, **kwargs):
        self.tokens = tokenize(edn, filename)
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
    def map(self, items: tuple): ...
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

class LiteralEDN(BuiltinEDN):
    R"""
    Round-tripping EDN parser using only types with literal notation.

    This form can be serialized with repr() and read back with
    ast.literal_eval(). It also easily serializes to JSON, but that
    format does not reliably distinguish ints from floats, so it may not
    round-trip when floats are integral. If this is a problem for your
    use case, override one of the methods.

    All collections read to tuples, with a prefix naming which.
    >>> [*LiteralEDN('#{1} [2 3] (4) {5 6, 7 8}').read()]
    [('set', 1), ('vector', 2, 3), ('list', 4), ('map', (5, 6), (7, 8))]

    Common primitives use the natural built-in types.
    >>> [*LiteralEDN('true false nil 0 0.0').read()]
    [True, False, None, 0, 0.0]

    The remaining atoms types render to strings with a prefix character.
    >>> [*LiteralEDN(R'"foo" foo :foo 0N 0.0M \x').read()]
    ['"foo', "'foo", ':foo', 'N0', 'M0.0', '\\x']

    Tags pass through.
    >>> [*LiteralEDN(R'#foo too #inst "1111-11-11"').read()]
    [('#foo', "'too"), ('#inst', '"1111-11-11')]
    """
    def __init__(self, edn, **kwargs):
        tags = dict(inst=partial(self.tag, 'inst'),
                    uuid=partial(self.tag, 'uuid'))
        super().__init__(edn, tags=tags, **kwargs)
    def set(self, elements: Iterator): return 'set', *elements
    def vector(self, elements: Iterator): return 'vector', *elements
    def list(self, elements: Iterator): return 'list', *elements
    def map(self, items: tuple): return 'map', *items
    def string(self, v: str): return '"' + v
    def symbol(self, v: str): return "'" + v
    def floatM(self, v: str): return "M" + v
    def intN(self, v: str): return "N" + v
    def char(self, v: str): return "\\" + v
    def tag(self, tag, element): return '#' + tag, element

class StandardEDN(BuiltinEDN):
    R"""Handles more cases, using only standard-library types.

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

    If the resulting list contains any non-pairs, it must have been
    read from a set. If the resulting list contains only pairs,
    it was likely read from a map, but could (in principle) have been
    read from a set of pairs, making it impossible to be sure. In the
    unlikely case this matters for your use, override cmap or cset to
    make them distinguishable.

    Symbol and keyword types map to `unittest.mock.sentinel`,
    a standard-library type meant for unit testing, but with the
    interning semantics desired for keywords: the same keyword always
    produces the same object. Using the same type for both is allowed
    by the spec, because they remain distinguishable by the leading
    character.
    >>> next(StandardEDN('[:foo foo]').read())
    (sentinel.:foo, sentinel.'foo)
    >>> _[0] is next(StandardEDN(':foo').read())
    True

    Chars get encoded to bytes.
    >>> next(StandardEDN(R'[\* "*" :* *]').read())
    (b'*', '*', sentinel.:*, sentinel.'*)

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
    without abiguity, but b'' has the advantage of being falsy,
    while still never comparing equal to any other standard EDN type.
    (Chars do render as bytes, but they are never length 0.)
    >>> next(StandardEDN('{0 0, 1 1, false 2, true 3}').read())
    {0: 0, 1: 1, b'': 2, sentinel.true: 3}

    The precision types are now distinguishable. A denominator-1
    fraction is a bit less natural. Python 2 used to have a separate
    int and long type, but now its int is arbitrary-precision, and it
    lacks a fixed-precision type. Because int is expected to be more
    common than intN, it gets the builtin, and intN gets something
    else.
    >>> next(StandardEDN('[0 0N 0M 0.0]').read())
    (0, Fraction(0, 1), Decimal('0'), 0.0)

    However, any numeric type of the same value still compares equal.
    Mixing numeric types in a hashtable is inadvisable, per the EDN
    spec, and ClojureScript has similar problems, so this is rarely
    an issue in practice.
    >>> next(StandardEDN('#{0 0N 0M 0.0}').read())
    frozenset({0})
    """
    set = frozenset
    vector = tuple
    floatM = Decimal
    # The above three are sensible choices, although Decimal is not in
    # builtins. The remainder are less natural, but work.
    cmap = cset = list
    char = staticmethod(str.encode)
    intN = Fraction  # Denominator 1.
    keyword = partial(getattr, sentinel)
    def symbol(self, v: str): return getattr(sentinel, "'"+v)
    nil = bool = {'false':b'', 'true':sentinel.true}.get
    tag = methodcaller  # Defers a call, but won't actually be a method.

class Box:
    """Wrapper to make keys behave like EDN.

    Python types have two issues representing EDN keys.

    First, EDN maps use different equality semantics. In Python,
    numbers are equal if their values are equal, even if they have
    different types (and bools are also numbers). In EDN, numbers
    must be of the same type to be equal. A Box is equal to an
    object only if the object is also of type Box, and their key
    objects are of exactly the same type and are equal.
    >>> Box(1) == Box(1)
    True
    >>> 1 == 1.0  # int and float can compare equal
    True
    >>> Box(1) == Box(1.0)  # But not when boxed.
    False

    (This is typically not an issue for numbers in practice as mixing
    numeric types like this is not recommended in EDN. ClojureScript
    also has trouble with this. Booleans are a different story.)

    Second, for most EDN collection types, the most natural Python
    analogue is unhashable. The result of custom tags may likewise be
    unhashable. If its key is unhashable, Box will fall back to
    using the hash of its type, a characteristic assumed to be
    immutable.
    >>> hash(Box({})) == hash(dict)
    True

    If a hashtable contains many Box keys, it may suffer
    degraded performance as keys cannot be dispersed over as many
    buckets when they produce equal hashes. This is likely no worse
    than the alternative of scanning through a list for a key,
    as they are at least dispersed by type, and any hashable boxed
    keys are also dispersed by value.

    Also, mutating anything used as a hashtable key is a bad idea,
    liable to cause surprises. Box enables this and can do nothing
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
    R"""Uses Box for keys.

    Unlike the simpler parsers, there are no cases expected to lose
    data. This a round-tripping parser.

    >>> next(BoxedEDN(R'{{1 1} 2, [2 2] 4}').read())
    {Box({Box(1): 1}): 2, Box([2, 2]): 4}
    >>> next(BoxedEDN(R'#{{1 1} 2 [2 2] 4}').read())
    frozenset({Box([2, 2]), Box(2), Box({Box(1): 1}), Box(4)})
    >>> len(next(BoxedEDN(R'#{0 0N 0M 0.0 false "0" \0}').read())) == 7
    True

    Due to the use of a non-standard type, unlike StandardEDN,
    unpickling the results in another environment may require a
    Garden of EDN install there.

    Bools use BuiltinEDN's method. They will get wrapped anyway,
    so there's no reason not to use the more natural types. Rendered
    types are otherwise as StandardEDN.
    >>> next(BoxedEDN(R'(0 0N 0M 0.0 false False)').read())
    (0, Fraction(0, 1), Decimal('0'), 0.0, False, sentinel.'False)
    """
    vector = list
    bool = BuiltinEDN.bool
    key = Box

class LilithHissp(BuiltinEDN):
    R"""Parses to Hissp. Allows Python programs to be written in EDN.

    The compiled output is standalone; LilithHissp compiles to
    standard-library Python; It doesn't necessitate any installation
    to run, beyond what is explicitly added to the program.
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

        Returns the compiled Python. Because forms are executed in
        turn, a form can use a macro defined previously in the same EDN.
        """
        try:
            self.compiler.evaluate = True
            return self.compiler.compile(self.read())
        finally:
            self.compiler.evaluate = False
    def __init__(self, edn, tags=(), *, qualname='<EDN>', ns=None, **kwargs):
        self.compiler = hissp.Compiler(qualname=qualname, ns=ns, evaluate=False)
        super().__init__(edn, tags, **kwargs)
    def string(self, v):
        return f'({repr(v)})'
    keyword = str
    floatM = Decimal
    def symbol(self, v):
        """
        Use a ``.`` for Hissp's ':', since that's not allowed in EDN.
        >>> print(LilithHissp('''
        ... (print 1 2 3 . sep .)
        ... ''').exec())
        1:2:3
        print(
          (1),
          (2),
          (3),
          sep=':')

        You can use ``/`` instead of ``..`` for fully-qualified imports.
        >>> print(LilithHissp('''
        ... (print math/tau math..pi)
        ... ''').exec())
        6.283185307179586 3.141592653589793
        print(
          __import__('math').tau,
          __import__('math').pi)

        Symbols use Lissp's munging rules.
        >>> next(LilithHissp('*%&?').read())
        'QzSTAR_QzPCENT_QzET_QzQUERY_'
        """
        if v == '.':
            return ':'
        if v != '/':
            v = v.replace('/', '..')
        return hissp.munge(v)
    def tag(self, tag, element):
        R"""

        Hissp's bundled prelude adds some basic utilities.
        >>> ns = {}  # You can re-use the same namespace.
        >>> LilithHissp('''
        ... (hissp/_macro_.prelude)
        ... ''', ns=ns).exec() and None

        #hissp/. is built in to LilithHissp & works like Lissp's inject.
        >>> [*LilithHissp(R'''
        ... "foo" ; Reads as a str containing a Python string literal.
        ... #hissp/. "foo" ; As str containing a Python identifier.
        ... foo ; Same.
        ... "40 + 2" ; Reads as str containing a Python string literal.
        ... #hissp/. "40 + 2" ; As str containing an add expression.
        ... #hissp/. (add 40 2) ; Read-time evaluation (to 42).
        ... #builtins/ord \* ; Use qualified unary func at read time.
        ... ''', ns=ns).read()]
        ["('foo')", 'foo', 'foo', "('40 + 2')", '40 + 2', 42, 42]

        Tags (like the #X for a unary function literal) fall back to
        the same read-time macros used by Lissp. For Lissp
        compatibility, the #X is interpreted as X# (i.e. XQzHASH_),
        found in the _macro_ object added by the prelude.
        >>> print(LilithHissp('''
        ... (print (#X #hissp/."X[::2]" "abc"))
        ... ''', ns=ns).exec())
        ac
        print(
          (lambda X:X[::2])(
            ('abc')))

        #hissp/! is built in to LilithHissp. It's used for read-time
        macros that take more than one argument. While Clojure has
        those, EDN tags are restricted to one argument. However,
        the argument can be a collection.

        #hissp/$ is also built in. It munges a string, making it act
        like a symbol. While EDN symbols are munged like Lissp,
        EDN does not allow certain characters in symbols that Lissp
        does. (Like @, because that's a built-in read-time macro in
        Clojure.)
        >>> next(LilithHissp(R'''
        ... #hissp/$"@"
        ... ''', ns=ns).read())
        'QzAT_'

        For example, Lissp's decorator read-time macro. Verbose,
        but works.
        >>> LilithHissp(R'''
        ... #hissp/!
        ... [#hissp/$"@" str.title
        ...  (define spam (quote spam))]
        ... (print spam)
        ... ''', ns=ns).exec() and None
        Spam

        For Lissp compatibility, primary argument comes last.
        >>> next(LilithHissp(R'''
        ... #hissp/!(builtins/print 1 2 . sep . 0)
        ... ''', ns=ns).read())
        0:1:2
        """
        extras = ()
        if tag == 'hissp/.':  # inject
            return eval(hissp.readerless(element, self.compiler.ns), self.compiler.ns)
        if tag == 'hissp/!':  # extra
            tag, *extras, element = element
        if tag == 'hissp/$':  # munge
            return hissp.munge(ast.literal_eval(element))
        *module, function = tag.replace('/', '..').split('..')
        if not module or re.match(rf"{MACROS}\.[^.]+$", function):
            function += hissp.munge('#')
        module = import_module(*module) if module else self.compiler.ns[MACROS]
        f = reduce(getattr, function.split('.'), module)
        args, kwargs = hissp.reader.parse_extras(extras)
        with self.compiler.macro_context():
            return f(element, *args, **kwargs)

class PyrMixin(AbstractEDN):
    """Mixin to make an EDN parser use Pyrsistent data structures.

    These fit EDN much better that Python's builtin collection types.
    """
    set = staticmethod(pset)
    map = staticmethod(pmap)
    list = staticmethod(plist)
    vector = pvector  # nondescriptor

class PyrBuiltinEDN(PyrMixin, BuiltinEDN):
    """Adds Pyrsistent collections to BuiltinEDN.

    >>> next(PyrBuiltinEDN(R'{[1] (1), #{2} 2N}').read())
    pmap({pset([2]): 2, pvector([1]): plist([1])})

    Unpickling the results in another environment requires
    Pyrsistent, but not Garden of EDN.
    """

class PyrStandardEDN(PyrMixin, StandardEDN):
    """Adds Pyrsistent collections to StandardEDN.

    >>> next(PyrStandardEDN(R'{[1] (1), #{2} 2N}').read())
    pmap({pset([2]): Fraction(2, 1), pvector([1]): plist([1])})

    Unpickling the results in another environment requires
    Pyrsistent, but not Garden of EDN.
    """

class PyrBoxedEDN(PyrMixin, BoxedEDN):
    """Adds Pyrsistent collections to BoxedEDN

    >>> next(PyrBoxedEDN(R'#{{} [] ()}').read())
    pset([Box(pmap({})), Box(plist([])), Box(pvector([]))])

    Pyrsistent collections are already hashable (if their elements
    are), but the equality problem remains and tags may still generate
    unhashable keys, so boxed keys are still required for a round-trip
    guarantee.
    >>> next(PyrBoxedEDN(R'#{0 0N 0M 0.0}').read())
    pset([Box(0), Box(Fraction(0, 1)), Box(Decimal('0')), Box(0.0)])
    >>> next(PyrStandardEDN(R'#{0 0N 0M 0.0}').read())
    pset([0])

    Unpickling the results in another environment requires
    Pyrsistent and Garden of EDN.
    """

class PandoraHissp(LilithHissp):
    R"""Interprets EDN colls as Pyrsistent collection except lists.

    Unlike LilithHissp, the compiled output is expected to typically
    require Pyrsistent to run; it is not standalone.

    Unlike Clojure (and EDN, typically), vectors, sets and maps read as
    construction expressions, not literally as the collections
    themselves. This approach is more compatible with Hissp.
    >>> for x in PandoraHissp('''
    ... [1 2] #{3} {4 5}
    ... ''').read():
    ...     print(x)
    ('pyrsistent..v', 1, 2)
    ('pyrsistent..s', 3)
    ('pyrsistent..pmap', ('hissp.._macro_.QzPCENT_', 4, 5))

    If you need the collections themselves at read time, use an inject.
    >>> for x in PandoraHissp('''
    ... #hissp/. [1 2] #hissp/. #{3} #hissp/. {4 5}
    ... ''').read():
    ...     print(x)
    pvector([1, 2])
    pset([3])
    pmap({4: 5})

    Beware that if Hissp is given a type of object without a literal
    syntax in Python, rather than the code to construct it, it must fall
    back to pickle in order to compile it all the way down to Python.
    >>> print(PandoraHissp('''
    ... #hissp/. [42]
    ... ''').exec())
    __import__('pickle').loads(  # pvector([42])
        b'cpvectorc\n'
        b'pvector\n'
        b'((lI42\n'
        b'atR.'
    )

    This can fail if the collection contains an unpicklable element.
    >>> PandoraHissp('''
    ... #hissp/. {1 (lambda .)}
    ... ''').exec() # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    hissp.compiler.CompileError:...
    (>   >  > >>pmap({1: <function <lambda> at 0x...>})<< <  <   <)
    # Compiler.pickle() PicklingError:
    #  Can't pickle <function <lambda> at 0x...

    But it's fine to pass these to another tag (or macro), as long as it
    doesn't compile all the way down. In this case, it reads to a None.
    >>> PandoraHissp('''
    ... #builtins/print #hissp/. {1 (lambda .)}
    ... ''').exec() # doctest: +ELLIPSIS
    pmap({1: <function <lambda> at 0x...>})
    'None'

    And, of course, the default construction expression works fine.
    >>> PandoraHissp('''
    ... (print {1 (lambda .)})
    ... ''').exec() and None # doctest: +ELLIPSIS
    pmap({1: <function <lambda> at 0x...>})
    """
    list = tuple
    def vector(self, elements): return 'pyrsistent..v', *elements
    def set(self, elements): return 'pyrsistent..s', *elements
    def map(self, items):
        return ('pyrsistent..pmap',
                ('hissp.._macro_.QzPCENT_',
                 *[x for kv in items for x in kv]))

class EDNImporter:
    def find_spec(self, fullname, path=None, target=None):
        filename = fullname.split('.')[-1] + '.edn'
        path = pathlib.Path(*path or '', filename)
        if path.is_file():
            return spec_from_loader(fullname, EDNLoader(fullname, str(path)))

class EDNLoader(FileLoader):
    def exec_module(self, module):
        module.__file__ = self.path
        path = pathlib.Path(self.path)
        self.edn = path.read_text()
        self.python = PandoraHissp(
            self.edn, filename=module.__file__, qualname=self.name, ns=vars(module)
        ).exec()
        return module
    def get_source(self, fullname):
        return self.edn

def __getattr__(name):
    '''Handles import actions that enable the use of PandoraHissp.

    Importing hooks allows the import of PandoraHissp EDN files.

    _this_file_as_main_ allows the python command to run an EDN file
    directly as main (implies hooks). It must be a valid EDN file
    that can also parse as Python which imports _this_file_as_main_
    from garden_of_edn, and must also contain a valid PandoraHissp
    program. For example::

        0 ; from garden_of_edn import _this_file_as_main_; """ "
        (print "Hello, World!")
        ;; """#"

    '''
    if name not in {'hooks', '_this_file_as_main_'}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import sys
    if EDNImporter not in map(type, sys.meta_path):
        sys.meta_path.append(EDNImporter())
    if name == '_this_file_as_main_':
        import inspect, sys
        __main__ = sys.modules["__main__"]
        source = inspect.getsource(__main__)
        PandoraHissp(
            source, filename=__main__.__file__, qualname='__main__', ns=vars(__main__)
        ).exec()
        raise SystemExit

if __name__ == '__main__':
    doctest.testmod()

# TODO: HisspEDN repl?
# TODO: basic pretty printer
# TODO: serializers?