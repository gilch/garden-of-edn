import ast
import builtins
import doctest
import re
from abc import ABCMeta, abstractmethod
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
    |(?P<_tag>\#[A-Za-z][-+.\d:#'!$%&*<=>?A-Z_a-z]*)
    |(?P<string>
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
    |(?P<keyword>:[-+.\d:#'!$%&*<=>?A-Z_a-z]+)
    |(?P<_symbol>[-+.]
                |(?:['!$%&*<=>?A-Z_a-z] # Always valid at start. Not [:#\d].
                    |[-+.][:#'!$%&*+\-.<=>?A-Z_a-z] # [-+.] can't be followed by a \d
                 )[-+.\d:#'!$%&*<=>?A-Z_a-z]*)
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

class BaseEDN(metaclass=ABCMeta):
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
        return self.set(self._parse_until('_rcub'))
    def _map(self, v):
        kvs = self._parse_until('_rcub')
        return self.map((k, next(kvs)) for k in kvs)
    def _list(self, v):
        return self.list(self._parse_until('_rpar'))
    def _vector(self, v):
        return self.vector(self._parse_until('_rsqb'))
    def _tag(self, v: str):
        return self.tags.get(v[1:], partial(self.tag, v[1:]))(next(self._parse()))
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
            if k!='discard':
                yield y
    # The remainder are meant for overrides.
    def tag(self, tag, v: str): raise KeyError(v)
    @abstractmethod
    def list(self, elements): ...
    def set(self, elements): return self.list(elements)
    def map(self, elements): return self.list(elements)
    def vector(self, elements): return self.list(elements)
    @abstractmethod
    def symbol(self, v: str): ...
    def string(self, v: str): return self.symbol(v)
    def keyword(self, v: str): return self.symbol(v)
    def bool(self, v: str): return self.symbol(v)
    def nil(self, v: str): return self.symbol(v)
    def float(self, v: str): return self.symbol(v)
    # The remainder don't fall back to symbol directly.
    def floatM(self, v: str): return self.float(v)
    def int(self, v: str): return self.float(v)
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
    of these use cases.
    >>> [*SimpleEDN.reads(R'"foo" :foo foo \x')]
    ['foo', ':foo', 'foo', 'x']

    If this is a problem for your use case, you can override one of the
    methods to return a different type.

    Collections simply use the Python collection with the same brackets.
    >>> [*SimpleEDN.reads(R'#{1}{2 3}(4)[5]')]
    [{1}, {2: 3}, (4,), [5]]

    EDN's collections are immutable, and are valid elements in EDN's set
    and as EDN's map keys, however, Python's builtin mutable collections
    (set, dict, and list) are unhashable, and therefore invalid in sets
    and as keys. In practice, most EDN data doesn't do this; keys are
    nearly always keywords or strings, but in those rare cases, this
    parser won't work.
    """
    set = set
    map = dict
    list = tuple
    vector = builtins.list
    def string(self, v):
        return ast.literal_eval(v.replace('\n',R'\n'))
    int = intN = int
    float = floatM = float
    keyword = symbol = str
    bool = {'false':False, 'true':True}.get
    def nil(self, v):
        return None

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

    Also see the docstrings for the float and symbol methods.
    """
    set = frozenset
    vector = tuple
    floatM = Decimal
    symbol = partial(getattr, sentinel)
    def symbol(self, v):
        """
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
        return getattr(sentinel, v)

class PyrEDN(AdvancedEDN):
    """EDN parser with Prysistent data structures.

    These fit EDN much better that Python's builtin collection types.
    Not guaranteed to round-trip."""
    set = staticmethod(pset)
    map = staticmethod(pmap)
    list = staticmethod(plist)
    vector = pvector  # nondescriptor
    bool = SimpleEDN.bool

class HisspEDN(PyrEDN):
    """Parses to Hissp. Allows Python programs to be written in EDN."""
    list = tuple
    def string(self, v):
        v = v.replace('\n',R'\n')
        v = ast.literal_eval(v)
        return f'({repr(v)})'
    keyword = symbol = str
    # bool = SimpleEDN.bool

if __name__ == '__main__':
    doctest.testmod()