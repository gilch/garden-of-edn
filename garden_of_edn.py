import ast
import builtins
import doctest
import re
from decimal import Decimal
from functools import partial
from itertools import takewhile
from unittest.mock import sentinel

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
    |(?P<_tag>\#[^\W\d_][-+.:#*!?$%&=<>\w]*)
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
      [-+.#*!?$%&=<>\w] # Second character cannot be :.
      [-+.:#*!?$%&=<>\w]*
     )
    |(?P<_symbol>
      [-+.]
      |(?:(?:[*!?$%&=<>]|[^\W\d]) # Always valid at start. Not [:#\d].
          |[-+.](?:[-+.:#*!?$%&=<>]|[^\W\d]) # [-+.] can't be followed by a \d
       )[-+.:#*!?$%&=<>\w]*)
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

class BaseEDN:
    """BaseEDN is highly customizable EDN parser.

    It is not an especially useful EDN parser in its own right, but does
    function. By default, all atoms render to strings and all
    collections render to tuples. This is intended as fallback behavior.
    Typical usage will override methods to render more specific types.
    """
    @classmethod
    def reads(cls, edn, tags=None):
        return cls(tokenize(edn), tags)._parse()
    def __init__(self, tokens, tags=None):
        self.tokens = tokens
        self.tags = tags or {}
    def _tokens_until(self, k):
        return takewhile(lambda kv: kv[0] != k, self.tokens)
    def _parse_until(self, k):
        return self._parse(self._tokens_until(k))
    def _discard(self, v):
        next(self._parse())
    def _set(self, v):
        elements = self._parse_until('_rcub')
        try:
            return self.set(elements)
        except TypeError:
            return self.cset(elements)
    def _map(self, v):
        ikvs = self._parse_until('_rcub')
        pairs = [(k,next(ikvs)) for k in ikvs]
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
        return ast.literal_eval(v.replace('\n',R'\n'))
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
    def tag(self, tag, v: str): raise KeyError(tag)
    list = tuple
    def vector(self, elements): return self.list(elements)
    def set(self, elements): return self.list(elements)
    def cset(self, elements): return self.list(elements)
    def map(self, elements): return self.list(elements)
    def cmap(self, elements): return self.list(elements)
    symbol = str
    def string(self, v: str): return self.symbol(v)
    def keyword(self, v: str): return self.symbol(v)
    def bool(self, v: str): return self.symbol(v)
    def nil(self, v: str): return self.symbol(v)
    def float(self, v: str): return self.symbol(v)
    # The remainder don't fall back to symbol directly.
    def floatM(self, v: str): return self.float(v)
    def int(self, v: str): return self.float(v)  # As ClojureScript.
    def intN(self, v: str): return self.int(v)
    def char(self, v: str): return self.string(v)

class SimpleEDN(BaseEDN):
    R"""Simple EDN parser.

    Renders each EDN type as the most natural equivalent Python type,
    making the resulting data easy to use from Python.

    The 20% solution for 80% of use cases. Does not implement the full
    EDN spec, but should have no trouble parsing a typical .edn config
    file.
    >>> [*SimpleEDN.reads(R'42 4.2 true nil')]
    [42, 4.2, True, None]

    However, this means it throws away information and can't round-trip
    back to the same EDN. Keywords, strings, symbols, and characters all
    become strings, because idiomatic Python uses the str type for all
    of these use cases. ClojureScript will also use strings for chars.
    >>> [*SimpleEDN.reads(R'"foo" :foo foo \x')]
    ['foo', ':foo', 'foo', 'x']

    If this is a problem for your use case, you can override one of the
    methods to return a different type.

    Mixing numeric type keys or set elements is inadvisable per the EDN
    spec, so this is rarely an issue in practice, but Python's equality
    semantics differ from EDN's: numeric types are equal when they
    have the same value (and bools are treated as 1-bit ints),
    regardless of type and precision. ClojureScript has a similar
    problem, treating all numbers as floats.
    >>> next(SimpleEDN(R'{false 1,0 2,0N 3,0.0 4,0M 5}'))
    {'x': 2, 0: 5}

    Collections simply use the Python collection with the same brackets.
    >>> [*SimpleEDN.reads(R'#{1}{2 3}(4)[5]')]
    [{1}, {2: 3}, (4,), [5]]

    EDN's collections are immutable, and are valid elements in EDN's set
    and as EDN's map keys, however, Python's builtin mutable collections
    (set, dict, and list) are unhashable, and therefore invalid in sets
    and as keys. In practice, most EDN data doesn't do this; keys are
    nearly always keywords or strings, but in those rare cases, this
    parser will fall back to tuples.
    """
    set = set
    map = dict
    list = tuple
    vector = builtins.list
    string = str
    int = int
    float = float
    symbol = str
    nil = bool = {'false':False, 'true':True}.get

class AdvancedEDN(SimpleEDN):
    """Handles more cases, using only standard-library types.

    But at the cost of being a little harder to use than SimpleEDN.
    EDN set and vector types now map to frozenset and tuple,
    respectively, which are hashable as long as their elements are,
    allowing them to be used as keys and in sets.
    >>> [*AdvancedEDN.reads('#{#{}} {[1 2] 3}')]
    [frozenset({frozenset()}), {(1, 2): 3}]

    This means that vectors and lists are no longer distinguishable, but
    (in practice) this distinction usually doesn't matter. Having two
    sequence type literals is somewhat useful in Clojure, but redundant
    in EDN. They compare equal when used in sets or as keys anyway.

    EDN map types still map to dict. While the `types.MappingProxyType`
    is an immutable view, it is still not hashable because the
    underlying mapping may still be mutable.

    Symbol and keyword types map to `unittest.mock.sentinel`, a
    standard-library type meant for unit testing, but with the
    interning semantics desired for keywords: the same keyword
    always produces the same object. Using the same type for these
    two is allowed by the spec, because they remain distinguishable
    by the leading colon.
    >>> next(AdvancedEDN.reads(':foo'))
    sentinel.:foo
    >>> _ is next(AdvancedEDN.reads(':foo'))
    True

    Python has a perfectly good bool type, but because EDN equality
    is different, it would cause collections to fail to round-trip
    in some cases.

    True is a special case of 1 and False 0 in Python, so the first
    values were overwritten.
    >>> next(SimpleEDN.reads('{0 0, 1 1, false 2, true 3}'))
    {0: 2, 1: 3}

    EDN doesn't consider these keys equal, so data was lost.
    AdvancedEDN can handle this without loss, by using the same
    sentinel types for booleans as well.
    >>> next(AdvancedEDN.reads('{0 0, 1 1, false 2, true 3}'))
    {0: 0, 1: 1, sentinel.false: 2, sentinel.true: 3}

    There is no abiguity with symbols, as ``false`` and ``true`` are
    always interpreted as booleans in EDN, so this can round-trip.
    Beware that ``sentinel.false`` is still truthy in Python.
    """
    set = frozenset
    vector = tuple
    floatM = Decimal
    symbol = partial(getattr, sentinel)
    nil = bool = {'false':b'', 'true':sentinel.true}.get

class PyrMixin:
    """Mixin to make an EDN parser use Pyrsistent data structures.

    These fit EDN much better that Python's builtin collection types.
    """
    set = staticmethod(pset)
    map = staticmethod(pmap)
    list = staticmethod(plist)
    vector = pvector  # nondescriptor

class SimplePyrEDN(PyrMixin, SimpleEDN):
    pass

class AdvancedPyrEDN(PyrMixin, AdvancedEDN):
    pass

class HisspEDN(AdvancedPyrEDN):
    """Parses to Hissp. Allows Python programs to be written in EDN."""
    list = tuple
    def string(self, v):
        v = v.replace('\n',R'\n')
        v = ast.literal_eval(v)
        return f'({repr(v)})'
    keyword = str
    bool = SimpleEDN.bool
    def symbol(self, v):
        if v=='&':
            return ':'
        return v

if __name__ == '__main__':
    doctest.testmod()