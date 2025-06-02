# Copyright 2022, 2023, 2024, 2025 Matthew Egan Odendahl
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""A Garden of EDN parsers and serializers. Bringing the Python to EDN.

The parsers are not validators. Behavior when given invalid EDN is
undefined. EDN is not especially well-specified to begin with.
"""
import ast
import builtins
import doctest
import io
import pathlib
import re
from abc import ABCMeta, abstractmethod
from collections import UserList, UserString
from collections.abc import Iterator, Iterable, Mapping, Sequence, Set
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, reduce
from importlib import import_module
from importlib.abc import FileLoader
from importlib.util import spec_from_loader
from itertools import takewhile
from operator import methodcaller
from typing import TextIO, NoReturn
from unittest.mock import sentinel
from uuid import UUID

import hissp
from hissp.compiler import MACROS
from pyrsistent import plist, pmap, pset, pvector
from pyrsistent._plist import _PListBase as PList

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

    However, this means it throws away information and can't always
    round-trip back to the same EDN data. Keywords, strings, symbols,
    and characters all become strings, because idiomatic Python uses the
    str type for all of these use cases. (ClojureScript will also use
    strings for chars.)
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

    "Round tripping" is in the sense of the original data, not the
    format of the text. It doesn't save whitespace formatting, comments,
    discarded items, or what string escapes may have been used.

    This form can be serialized with repr() and read back with
    ast.literal_eval(). It also easily serializes to JSON, but that
    format does not reliably distinguish ints from floats, so JSON may
    not round-trip when floats are integral. If this is a problem for
    your use case, override one of the methods.

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
    seq literals is somewhat useful in Clojure, but redundant in EDN.
    They compare equal when used in maps or sets anyway. If this matters
    for your use case, you can override one of them.

    EDN map types still render to dict. No hashable mapping type is
    available in the standard library. While the
    `types.MappingProxyType` is an immutable view, it is still not
    hashable because the underlying mapping may still be mutable.

    List is still used as a fallback for unhashable elements.
    There's not much point in using a tuple here, since a tuple with
    an unhashable element is itself unhashable.

    >>> a_set = next(StandardEDN(R'#{{1 1} 2 {2 2} 4}').read())
    >>> type(a_set)
    <class 'list'>
    >>> a_set
    [{1: 1}, 2, {2: 2}, 4]

    Python's `in` operator works on any collection type, and set
    operations are not hard to do with list comprehensions:
    >>> {1: 1} in a_set
    True
    >>> [x for x in a_set if x in [2,4,6]]  # Intersection
    [2, 4]
    >>> [x for x in a_set if x not in [2,4,6]]  # Difference
    [{1: 1}, {2: 2}]
    >>> _ + [2,4,6]  # Union (without duplicates after difference)
    [{1: 1}, {2: 2}, 2, 4, 6]

    >>> a_map = next(StandardEDN(R'{{1 1} 2, {2 2} 4}').read())
    >>> a_map
    [({1: 1}, 2), ({2: 2}, 4)]

    If the resulting list contains only pairs, it was likely read from a
    map, but could (in principle) have been read from a set of pairs.
    Check for the UserList type to distigish these cases.
    >>> type(a_map)
    <class 'collections.UserList'>

    An iterable of pairs is already the form a dict's .items() would
    take. Lookups can be done by scanning:
    >>> next(v for k, v in a_map if k=={2: 2})
    4

    An absent item would result in a StopIteration rather than a
    KeyError, but next() can have a default like dict.get()
    >>> next((v for k, v in a_map if k==7), 0)
    0

    Symbol and keyword types map to `unittest.mock.sentinel`,
    a standard-library type meant for unit testing, but with the
    interning semantics desired for keywords: the same keyword always
    produces the same object. Using the same type for both is allowed
    by the spec, because they remain distinguishable by the leading
    character.
    >>> next(StandardEDN('[:foo foo]').read())
    (sentinel.:foo, sentinel.foo)
    >>> _[0] is next(StandardEDN(':foo').read())
    True

    Chars become length-1 strings as in ClojureScript.
    >>> next(StandardEDN(R'[\* "*" :* *]').read())
    ('*', '*', sentinel.:*, sentinel.*)

    They're distinguishable as UserStrings rather than the builtin str:
    >>> type(_[0])
    <class 'collections.UserString'>

    Python has a perfectly good bool type, but because EDN equality
    is different, it would cause collections to fail to round-trip
    in some cases.

    True is a special case of 1 and False 0 in Python, so the first
    values here were overwritten:
    >>> next(BuiltinEDN('{0 0, 1 1, false 2, true 3}').read())
    {0: 2, 1: 3}

    EDN doesn't consider these keys equal, so data was lost.
    StandardEDN can handle this without loss, by using the same
    sentinel type for true as well. sentinel.false could also be used
    without ambiguity, but b'' has the advantage of being falsy,
    while still never comparing equal to any other StandardEDN type.
    >>> next(StandardEDN('{0 0, 1 1, false 2, true 3}').read())
    {0: 0, 1: 1, b'': 2, sentinel.true: 3}

    The precision types are now distinguishable. A denominator-1
    fraction is a bit less natural. Python (in version 2) used to have a
    separate int and long type, but now its int is arbitrary-precision,
    and it lacks a fixed-precision type. Because int is expected to be
    more common than intN, it gets the builtin, and intN gets something
    else.
    >>> next(StandardEDN('[0 0N 0M 0.0]').read())
    (0, Fraction(0, 1), Decimal('0'), 0.0)

    However, any numeric type of the same value still compares equal.
    Mixing numeric types in a hashtable is inadvisable, per the EDN
    spec, and ClojureScript has similar problems, so this is rarely
    an issue in practice.
    >>> next(StandardEDN('#{0 0N 0M 0.0}').read())
    frozenset({0})

    There's a similar problem with chars and length-1 strings, which
    Python considers equal. EDN should technically not consider them
    equal, and it's not specifically called out in the spec (unlike
    the numbers), but ClojureScript has the same problem, so this is
    also rarely an issue in practice.
    >>> next(StandardEDN(R'#{\a "a"}').read())
    frozenset({'a'})
    """
    # TODO: doctest tag/methodcaller examples
    set = frozenset
    vector = tuple
    floatM = Decimal
    # The above three are sensible choices, although Decimal is not in
    # builtins. The remainder are less natural, but work.
    cmap = UserList
    cset = list
    char = UserString
    intN = Fraction  # Denominator defaults to 1 if passed an int.
    symbol = keyword = staticmethod(partial(getattr, sentinel))
    nil = bool = {'false': b'', 'true': sentinel.true}.get
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
    data. This a round-tripping parser (at the level of data, not text).

    >>> next(BoxedEDN(R'{{1 1} 2, [2 2] 4}').read())
    {Box({Box(1): 1}): 2, Box([2, 2]): 4}
    >>> next(BoxedEDN(R'#{{1 1} 2 [2 2] 4}').read())
    frozenset({Box([2, 2]), Box(2), Box(4), Box({Box(1): 1})})
    >>> len(next(BoxedEDN(R'#{0 0N 0M 0.0 false "0" \0}').read())) == 7
    True

    Due to the use of a non-standard type, unlike StandardEDN,
    unpickling the results in another environment may require a
    Garden of EDN install there.

    Bools use BuiltinEDN's method. They will get wrapped anyway,
    so there's no reason not to use the more natural types. Rendered
    types are otherwise as StandardEDN.
    >>> next(BoxedEDN(R'(0 0N 0M 0.0 false False)').read())
    (0, Fraction(0, 1), Decimal('0'), 0.0, False, sentinel.False)
    """
    vector = list
    bool = BuiltinEDN.bool
    key = Box

# TODO: figure out aliases
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
        self.compiler.evaluate = True
        try:
            return self.compiler.compile(self.read())
        finally:
            self.compiler.evaluate = False
    def __init__(self, edn, tags=(), *, qualname='<EDN>', env=None, **kwargs):
        self.compiler = hissp.Compiler(qualname=qualname, env=env, evaluate=False)
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
        >>> env = {}  # You can re-use the same environment.
        >>> LilithHissp('''
        ... (hissp/_macro_.prelude)
        ... ''', env=env).exec() and None

        #hissp/. is built into LilithHissp & works like Lissp's inject.
        >>> [*LilithHissp(R'''
        ... "foo" ; Reads as a str containing a Python string literal.
        ... #hissp/. "foo" ; As str containing a Python identifier.
        ... foo ; Same.
        ... "40 + 2" ; Reads as str containing a Python string literal.
        ... #hissp/. "40 + 2" ; As str containing an add expression.
        ... #hissp/. (add 40 2) ; Read-time evaluation (to 42).
        ... #builtins/ord \* ; Use qualified unary func at read time.
        ... ''', env=env).read()]
        ["('foo')", 'foo', 'foo', "('40 + 2')", '40 + 2', 42, 42]

        Tags (like the #X for a unary function literal) fall back to
        the same read-time macros used by Lissp. For Lissp
        compatibility, the #X is interpreted as X# (i.e. XQzHASH_),
        found in the _macro_ object added by the prelude.
        >>> print(LilithHissp('''
        ... (print (#X #hissp/."X[::2]" "abc"))
        ... ''', env=env).exec())
        ac
        print(
          (lambda X: X[::2])(
            ('abc')))

        #hissp/$ is also built in. It munges a string, making it act
        like a symbol. While EDN symbols are munged like Lissp,
        EDN does not allow certain characters in symbols that Lissp
        does. (Like @, because that's a built-in read-time macro in
        Clojure.)
        >>> next(LilithHissp(R'''
        ... #hissp/$"@"
        ... ''', env=env).read())
        'QzAT_'
        """
        if tag == 'q': return 'quote', element
        # TODO: quasiquotes?
        if tag == 'hissp/.':  # inject
            return eval(hissp.readerless(element, self.compiler.env), self.compiler.env)
        if tag == 'hissp/$':  # munge
            return hissp.munge(ast.literal_eval(element))
        *module, function = tag.replace('/', '..').split('..')
        if not module or re.match(rf"{MACROS}\.[^.]+$", function):
            function += hissp.munge('#')
        module = import_module(*module) if module else self.compiler.env[MACROS]
        f = reduce(getattr, function.split('.'), module)
        with hissp.compiler.macro_context(self.compiler.env):
            return f(element)

class PyrMixin(AbstractEDN):
    """Mixin to make an EDN parser use Pyrsistent data structures.

    These fit EDN much better that Python's builtin collection types.
    """
    set = staticmethod(pset)
    map = staticmethod(pmap)
    list = staticmethod(plist)
    vector = staticmethod(pvector)

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
    pset([Box(pmap({})), Box(pvector([])), Box(plist([]))])

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


class _SExpression:
    R"""
    Helper object to make unambiguous data for EDN.

    Common simple cases.
    >>> S.foo  # symbol type
    S@'foo'
    >>> +S.foo  # keywords are still of symbol type
    S@':foo'
    >>> S.nil  # a particular symbol instance
    S@'nil'
    >>> S.true # also a particular symbol instance
    S@'true'
    >>> S/"a"  # a character type
    S/'a'
    >>> type(S/"a").__name__
    'Char'
    >>> S@42  # IntN type forces N even for small ints
    S@'42N'

    S.false and S.nil are falsy, but not ints.
    >>> bool(S.nil)
    False
    >>> bool(S.false)
    False
    >>> S.false == 0
    False
    >>> S.true == 1
    False

    But an IntN is an int.
    >>> S@42 + 1
    43
    >>> str(S@42)
    '42N'

    >>> S@"foo!"  # String must contain an EDN atom.
    S@'foo!'
    >>> S.foo.bar
    S@'foo.bar'
    >>> type(_).__name__
    'Symbol'
    >>> S.foo@"bar!"@"baz!"  # appends segments
    S@'foo.bar!.baz!'
    >>> S.foo is S.foo  # symbol interning
    True
    >>> S.foo.bar/"baz"  # namespaces
    S@'foo.bar/baz'
    >>> S@"foo!.bar"/S.baz
    S@'foo!.bar/baz'

    Symbols convert to Keywords with +.
    >>> +S.foo.bar/S.baz
    S@':foo.bar/baz'

    Returns standard-library types in some cases:
    >>> S@"42"  # EDN integer to Python int
    42
    >>> S@"4.2"  # EDN float to Python float
    4.2
    >>> S@"4.2M"  # EDN floatM to Python Decimal
    Decimal('4.2')
    >>> S@'"foo"'  # EDN string to Python str
    'foo'

    Makes pmaps and psets even if types are unhashable (via Box).
    >>> S[1:2, 3:4]
    pmap({Box(1): 2, Box(3): 4})
    >>> S@[1,2,3]
    pset([Box(1), Box(2), Box(3)])

    Also makes plist.
    >>> S(1,2,3)
    plist([1, 2, 3])
    """
    # TODO: figure out pvector?
    def __matmul__(self, other):
        match other:
            case int(): return IntN(other)
            case str(): return Atom.from_edn(other)
        return pset(map(Box, other))
        # return pvector(other)
    def __truediv__(self, other):
        return Char(other)
    def __getattr__(self, item):
        return Symbol(item)
    def __getitem__(self, item):
        return pmap({Box(s.start): s.stop for s in item}, len(item))
    def __call__(self, *args):
        return plist(args)
S = _SExpression()

class Atom:
    """Represents a single atom of EDN data. Normally made using S."""
    @classmethod
    def from_edn(cls, atom):
        """Besides an instance of one of the Atom subclasses,
        may return an int, float, str, or Decimal object as appropriate.
        """
        first, *rest = LiteralEDN(atom).read()
        if rest or type(first) is tuple:
            raise ValueError(f'Expected atom, got {atom}')
        match first:
            case bool() | None:   return Symbol(atom)
            case int() | float(): return first
            case str():
                match first[0]:
                    case '"': return first[1:]
                    case "'": return Symbol(first[1:])
                    case ':': return Symbol(first)
                    case 'N': return IntN(first[1:])
                    case 'M': return Decimal(first[1:])
                    case '\\': return Char(first[1:])
        # TODO: assert_never(v) in 3.11
        raise AssertionError('unreachable', atom)
    def __repr__(self):
        return f'S@{str(self)!r}'

class IntN(int, Atom):
    """An int subclass representing an arbitrary-precision EDN int."""
    def __str__(self):
        return super().__repr__() + 'N'
    __repr__ = Atom.__repr__

class Char(Atom):
    """Represents a single EDN character, and isn't a string."""
    def __init__(self, c):
        if len(c) != 1:
            raise ValueError(f'Expected character, got {c}')
        self.c = c
    def __str__(self):
        return self.c
    def __repr__(self):
        return f'S/{self.c!r}'

class Symbol(Atom):
    """Represents an EDN nil, true, false, symbol, or keyword."""
    _interns = {}
    def __hash__(self):
        return hash(self.name)
    def __bool__(self):
        return not self in {S.false, S.nil}
    def __new__(cls, name):
        new = object.__new__(cls)
        new.name = name
        return cls._interns.setdefault(name, new)
    def __getattr__(self, item):
        return Symbol(f'{self.name}.{item}')
    def __pos__(self):
        return Symbol(':' + self.name)
    def __truediv__(self, other):
        if isinstance(other, str):
            other = S@other
        if "/" in self.name or "/" in other.name:
            raise ValueError("Only one / allowed in EDN symbols.")
        return S@f'{self.name}/{other.name}'
    def __matmul__(self, other):
        if isinstance(other, Symbol):
            other = other.name
        return S@f'{self.name}.{other}'
    def __str__(self):
        return self.name

class GardenEDN(PyrMixin):
    """Uses the Atom types, Box, and Pyrsistent collections to
    follow EDN as closely as possible. This is a round-tripping parser.

    Due to the use of non-standard types, unpickling the results in
    another environment may require a Garden of EDN install there.
    """
    symbol = string = keyword = bool = nil = float = floatM = int = intN = char = (
        Atom.from_edn
    )
    key = Box

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
    # pvector([42])
    __import__('pickle').loads(b'cpyrsistent._pvector\npython_pvector\n((lI42\natR.')

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
            self.edn, filename=module.__file__, qualname=self.name, env=vars(module)
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
        try:
            PandoraHissp(
                source, filename=__main__.__file__, qualname='__main__', env=vars(__main__)
            ).exec()
        except:
            import traceback
            traceback.print_exc()
            raise
        raise SystemExit

class AsEDN:
    """
    EDN serializer base class.

    Subclasses should override the dispatch() method which
    should handle what it knows call super().dispatch(obj) for the rest.
    If you'd instead prefer to avoid the default cases, insert the
    GiveUpMixin in the appropriate place in the method resolution order,
    and call super().dispatch(obj) anyway, to cooperate with further
    subclasses.
    """
    def __init__(self, *, comma=''):
        self.context = []
        self.comma = comma
    def dispatch(self, obj) -> Iterable[str]:
        R"""
        Handles only the most obvious cases.

        Tags:
        >>> AsEDN.dumps([datetime.fromisoformat('2025-05-04T13:13:13')])
        '#inst,"2025-05-04T13:13:13"'
        >>> AsEDN.dumps([UUID(int=0)])
        '#uuid,"00000000-0000-0000-0000-000000000000"'

        Simple atoms:
        >>> AsEDN.dumps([None, True, Decimal('0.070'), 4.2, 'Hi'])
        'nil,true,0.070M,4.2,"Hi"'

        Ints get an N only if outside the signed 64-bit range:
        >>> AsEDN.dumps([42, 2**63])
        '42,9223372036854775808N'

        >>> print(AsEDN.pdumps([{frozenset({0,1}):iter('ab'), Box({'foo':(1,2)}): [1,2]}]))
        {#{0
           1}
         ("a"
          "b")
         ,
         {"foo"
          [1
           2]}
         [1
          2]}
        <BLANKLINE>

        Also recognizes the Atom subclasses:
        >>> AsEDN.dumps([S.foo, +S.foo, S.nil, S.false, S/"\n", S@42])
        'foo,:foo,nil,false,\\newline,42N'
        """
        match obj:
            case Box():      return self.dispatch(obj.k)
            # atoms
            case Symbol() | IntN(): return str(obj)
            case Char():     return self.char(str(obj))
            case datetime(): return '#inst', f'"{obj.isoformat()}"'
            case UUID():     return '#uuid', f'"{obj}"'
            case None:       return self.nil()
            case bool():     return self.bool(obj)
            case Decimal():  return self.floatM(obj)
            case float():    return self.float(obj)
            case str():      return self.string(obj)
            case int():
                # The first < is correct. Java longs include -2**63, but
                # C doesn't guarantee it, so it needs the N.
                return self.int(obj) if -2**63 < obj < 2**63 else self.intN(obj)
            # collections
            case Mapping():  return self.map(obj.items())
            case Set():      return self.set(obj)
            case Sequence() if not isinstance(obj, (PList, bytes, str)):
                return self.vector(obj)
            case Iterable() if not isinstance(obj, (bytes, str)):
                return self.list(obj)
            case _: raise TypeError(f"No handler for {obj!r}.")
    @classmethod
    def dumps(cls, objects: Iterable, *, sep=','):
        """Quick dump to string separating tokens with sep."""
        return sep.join(cls().node(objects))
    @classmethod
    def dump(cls, obj, fp: TextIO, *, sep=','):
        """Quick dump to file separating tokens with sep.

        Ugly, but computers don't care. Users with an EDN-aware editor
        can easily auto-format it to something more readable.
        """
        for token in cls().node([obj]):
            fp.write(token)
            fp.write(sep)
    @classmethod
    def pdump(cls, obj, fp: TextIO):
        """Minimal "pretty" EDN dumper.

        You're better off piping `dumps()` through a quality EDN
        formatter, but this is useful for getting a human-readable
        output for debugging, etc. when you don't have a better one
        handy.
        """
        self = cls(comma=',')
        is_after_open = True
        for token in iter(self.node([obj])):
            if not is_after_open and token not in {']', '}', ')'}:
                fp.write('\n' + len(''.join(self.context)) * ' ')
            fp.write(token)
            is_after_open = token in {'(', '#{', '{', '['}
        fp.write('\n')
    @classmethod
    def pdumps(cls, objects: Iterable) -> str:
        """Convenience StringIO interface for pdump()."""
        self = cls()
        sio = io.StringIO()
        for obj in objects:
            self.pdump(obj, sio)
        sio.seek(0)
        return sio.read()
    def node(self, children) -> Iterable[str]:
        for child in children:
            element = self.dispatch(child)
            if isinstance(element, str):
                yield element
            else:
                yield from element
    def seq(self, open, elements, close):
        yield open
        self.context.append(open)
        yield from elements
        yield close
        self.context.pop()
    def list(self, elements: Iterable) -> Iterable[str]:
        return self.seq('(', self.node(elements), ')')
    def vector(self, elements: Iterable) -> Iterable[str]:
        return self.seq('[', self.node(elements), ']')
    def set(self, elements: Iterable) -> Iterable[str]:
        return self.seq('#{', self.node(elements), '}')
    def map(self, pairs: Iterable) -> Iterable[str]:
        return self.seq('{', self.pairs(pairs), '}')
    def pairs(self, kvs, _sep=''):
        for kv in kvs:
            yield from _sep
            yield from self.node(kv)
            _sep = self.comma
    def tag(self, tag: str, v) -> Iterable[str]:
        yield f'#{tag}'
        yield from self.node([v])
    @staticmethod
    def symbol(v: str) -> str: return v
    @staticmethod
    def string(v: str) -> str: return f'"{v.replace("\\", "\\\\").replace('"', R'\"')}"'
    @staticmethod
    def keyword(k: str) -> str: return f':{k}'
    @staticmethod
    def bool(v: bool) -> str: return 'true' if v else 'false'
    @staticmethod
    def nil() -> str: return 'nil'
    @staticmethod
    def float(v: float) -> str: return str(v)
    @staticmethod
    def floatM(v: Decimal) -> str: return f'{v}M'
    @staticmethod
    def intN(v: int) -> str: return f'{v}N'
    @staticmethod
    def int(v: int) -> str: return str(v)
    @staticmethod
    def char(
        v: str, _chars={'\n': 'newline', '\r': 'return', ' ': 'space', '\t': 'tab'}
    ) -> str:
        if len(v) != 1:
            raise ValueError(f'Expected char, got {v!r}.')
        return R'\{}'.format(
            _chars.get(v, v)
        )

class GiveUpMixin(AsEDN):
    """
    Overrides dispatch() to raise a TypeError and not call the super()
    version. Used to suppress any further inherited handlers, for cases
    when unexpected types should raise errors.
    """
    @abstractmethod
    def dispatch(self, obj) -> NoReturn:
        """Always raises a TypeError."""
        raise TypeError(f"No handler for {obj!r}.")

class StrAsEDN(AsEDN):
    """
    Overrides the str handling to uncritically emit their contents.
    This can still be used to emit EDN strings, by using a str
    containing an EDN string.
    """
    def dispatch(self, obj) -> Iterable[str]:
        R"""Overrides the str handler to return its contents:

        >>> edn = [':spam', '42N', 'eggs', '"sausage"', R'\newline']
        >>> print(StrAsEDN.dumps(edn, sep=', '))
        :spam, 42N, eggs, "sausage", \newline

        Strings are expected to contain valid EDN, but this is not
        verified or secure. Not recommended for untrusted data.
        Behavior is undefined on invalid input and may or may not result
        in valid EDN output.

        Calls superclass method if obj is not a str.
        """
        match obj:
            case str(): return obj
            case _: return super().dispatch(obj)

class TupleAsEDNList(AsEDN):
    """
    Overrides tuple handling to emit EDN lists (instead of vectors).
    Given the abstract default handlers, this is effectively the inverse
    of BuiltinEDN, which doesn't claim to round-trip.

    Other Sequence types (besides PList, bytes, and str) sill serialize
    as vectors by default.
    """
    def dispatch(self, obj) -> Iterable[str]:
        """
        >>> TupleAsEDNList.dumps([(1,2,3), [1,2,3]])
        '(,1,2,3,),[,1,2,3,]'
        """
        match obj:
            case tuple(): return self.list(obj)
            case _: return super().dispatch(obj)

_SentinelObject = type(sentinel.X)
class StandardAsEDN(AsEDN):
    """
    The inverse of the StandardEDN parser, given the abstract default
    handlers, which would also invert the BoxedEDN, PyrStandardEDN,
    and PyrBoxedEDN cases.
    """
    def dispatch(self, obj) -> Iterable[str]:
        R"""
        >>> s = sentinel
        >>> edn = [b'', Fraction(2/1), s.true, UserString('\n'), getattr(s, ":spam"), s.spam]
        >>> print(StandardAsEDN.dumps(edn))
        false,2N,true,\newline,:spam,spam
        """
        match obj:
            case b'':                           return self.bool(False)
            case Fraction():                    return self.intN(int(obj))
            case sentinel.true:                 return self.bool(True)
            case UserString() if len(obj) == 1: return self.char(obj)
            case _SentinelObject():             return self.symbol(obj.name)
            case _: return super().dispatch(obj)

class LiteralAsEDN(AsEDN):
    """
    Round-tripping serializer using the LiteralEDN format.
    It will also fall back to the superclass handlers.
    """
    def dispatch(self, obj) -> Iterable[str]:
        R"""Serializes the EDN encoding produced by LiteralEDN.

        >>> edn = [('map',(('set','\\\t',':spam'),'N12')
        ...              ,(('list','M1',"'eggs"),('vector',1,'"sausage')))]
        >>> print(LiteralAsEDN.pdumps(edn))
        {#{\tab
           :spam}
         12N
         ,
         (1M
          eggs)
         [1
          "sausage"]}
        <BLANKLINE>
        """
        match obj:
            # collections
            case ['set', *xs] if type(obj) is tuple:    return self.set(xs)
            case ['vector', *xs] if type(obj) is tuple: return self.vector(xs)
            case ['list', *xs] if type(obj) is tuple:   return self.list(xs)
            case ['map', *kvs] if type(obj) is tuple:   return self.map(kvs)
            # primitives (AsEDN has these, but see LiteralOnlyAsEDN.)
            case None:    return self.nil()
            case bool():  return self.bool(obj)
            case int():   return self.int(obj)
            case float(): return self.float(obj)
            # str codes
            case str(v) if v and v[0] in R'''\"':NM''':
                c, cs = v[0], v[1:]
                match c:
                    case '\\': return self.char(cs)
                    case '"': return self.string(cs)
                    case "'": return self.symbol(cs)
                    case ":": return self.keyword(cs)
                    case "N": return self.intN(int(cs))
                    case "M": return self.floatM(Decimal(cs))
                raise AssertionError('unreachable', v)  # TODO: assert_never(v) in 3.11
            # tags pass through
            case [str(tag), v] if type(obj) is tuple and tag.startswith('#'):
                return self.tag(tag[1:], v)
            case _: return super().dispatch(obj)

class LiteralOnlyAsEDN(LiteralAsEDN, GiveUpMixin): pass

class GardenAsEDN(TupleAsEDNList):
    def dispatch(self, obj) -> Iterable[str]:
        match obj:
            case Symbol() | IntN():
                return str(obj)
            case Char():
                return self.char(str(obj))
            case _: return super().dispatch(obj)

def minify(edn: str) -> str:
    """Rewrites EDN whitespace to a single ',' between each token
    via AsEDN.dumps().
    """
    return LiteralOnlyAsEDN.dumps(LiteralEDN(edn).read())

def pformat(edn: str) -> str:
    """Convenience function to pretty-format EDN
    via AsEDN.pdumps().
    """
    return LiteralOnlyAsEDN.pdumps(LiteralEDN(edn).read())

def pprint(edn: str) -> None:
    """Convenience function to pretty-print EDN via pformat()."""
    print(pformat(edn))

if __name__ == '__main__':
    doctest.testmod()

# TODO: HisspEDN repl? Via nrepl?
# TODO: Hypothesis tests?