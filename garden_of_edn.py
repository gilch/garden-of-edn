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
    |(?P<char>\\(?:newline|return|space|tab|u[\dA-Fa-f]{4}|\S))
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
    def edn(cls, edn, tags={}):
        return cls(tokenize(edn), tags)
    @classmethod
    def reads(cls, edn, tags={}):
        return cls.edn(edn, tags).parse()
    def __init__(self, tokens, tags={}):
        self.tokens = tokens
        self.tags = tags
    def parse(self, tokens=None):
        for k, v in tokens or self.tokens:
            y = getattr(self, k)(v)
            if k!='discard':
                yield y
    def _tokens_until(self, k):
        return takewhile(lambda kv: kv[0] != k, self.tokens)
    def _parse_until(self, k):
        return self.parse(self._tokens_until(k))
    def _discard(self, v):
        next(self.parse())
    def _set(self, v):
        return self.set(self._parse_until('_rcub'))
    @abstractmethod
    def set(self, elements):
        ...
    def _map(self, v):
        kvs = self._parse_until('_rcub')
        return self.map([k, next(kvs)] for k in kvs)
    @abstractmethod
    def map(self, elements):
        ...
    def _list(self, v):
        return self.list(self._parse_until('_rpar'))
    @abstractmethod
    def list(self, elements):
        ...
    def _vector(self, v):
        return self.vector(self._parse_until('_rsqb'))
    @abstractmethod
    def vector(self, elements):
        ...
    def _tag(self, v: str):
        return self.tags.get(v[1:], partial(self.tag, v[1:]))(next(self.parse()))
    @abstractmethod
    def tag(self, tag, v: str):
        ...
    @abstractmethod
    def string(self, v: str):
        ...
    def _int(self, v: str):
        return self.intN(v[:-1]) if v.endswith('N') else self.int(v)
    @abstractmethod
    def int(self, v: str):
        ...
    @abstractmethod
    def intN(self, v: str):
        ...
    def _float(self, v: str):
        return self.floatM(v[:-1]) if v.endswith('M') else self.float(v)
    @abstractmethod
    def float(self, v: str):
        ...
    @abstractmethod
    def floatM(self, v: str):
        ...
    @abstractmethod
    def keyword(self, v: str):
        ...
    @abstractmethod
    def symbol(self, v: str):
        ...
    @abstractmethod
    def bool(self, v: str):
        ...
    @abstractmethod
    def nil(self, v: str):
        ...
    @abstractmethod
    def char(self, v: str):
        ...

class SimpleEDN(BaseEDN):
    R"""Simple EDN parser.

    The 20% solution for 80% of use cases. Does not implement the full
    EDN spec, but should have no trouble parsing a typical .edn config
    file, and renders each EDN type as the most natural equivalent
    Python type, making the resulting data easy to use from Python.
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
    def tag(self, tag, v):
        raise ValueError(f'Unknown tag {tag}')
    def string(self, v):
        return ast.literal_eval(v.replace('\n',R'\n'))
    int = intN = int
    float = floatM = float
    keyword = symbol = str
    bool = {'false':False, 'true':True}.get
    def nil(self, v):
        return None
    def char(self, v):
        v = v[1:]
        v = {'newline':'\n','return':'\r','space':' ','tab':'\t'}.get(v,v)
        if v.startswith('u'):
            v = ast.literal_eval(Rf"'\{v}'")
        return v

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
    bool = keyword = symbol = partial(getattr, sentinel)
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
    bool = keyword = symbol

class LiteralEDN(SimpleEDN):
    """Round-tripping EDN parser, targeting only Python literal types.

    Parses data that might not round-trip as Ellipsis groups.
    EDN lists still map to tuples. This is unambiguous because Ellipsis
    is not an EDN type, therefore any tuple beginning with "..." does
    not represent an EDN list. The second element is the equivalent
    Python type, and may be followed by additional elements.

    This is an EDN parser only, not a validator. Behavior when passed
    invalid EDN is undefined.

    Python literal notation is the data language parsed by
    `ast.literal_eval`. Python literal types are the type expressible in
    that language. The results of this parser could also be expressed in
    Python literal notation.
    """
    def set(self, elements):
        """Parses an EDN set as an Ellipsis group containing a set,
        followed by any additional elements of unhashable types meant to
        be in the EDN set.

        >>> next(LiteralEDN.reads('#{1 [] #{} {}}'))
        (Ellipsis, {1}, [], (Ellipsis, set()), (Ellipsis, {}))
        """
        s, L = set(), []
        for e in elements:
            try:
                hash(e)
            except TypeError:
                L.append(e)
            else:
                s.add(e)
        return ..., s, *L
    def map(self, items):
        """Parses an EDN map as an Ellipsis group containing a dict,
        followed by any items with unhashable keys meant to be in the
        map.

        >>> next(LiteralEDN.reads('{1 2, [] 3, #{} 4}'))
        (Ellipsis, {1: 2}, [[], 3], [(Ellipsis, set()), 4])
        """
        d, kvs = {}, []
        for k, v in items:
            try:
                hash(k)
            except TypeError:
                kvs.append([k, v])
            else:
                d[k] = v
        return ..., d, *kvs
    def int(self, v):
        """Parses an EDN int as an int and a precise EDN int (N suffix)
        as an Ellipsis group containing an int.

        >>> [*LiteralEDN.reads('42 42N')]
        [42, (Ellipsis, 42)]
        """
        if v.endswith('N'):
            return ..., int(v[:-1])
        return int(v)
    def float(self, v):
        """Parses an EDN float as an Ellipsis group containing a float.

        If precise (M suffix), also contains a string.
        >>> [*LiteralEDN.reads('.42 .42M')]
        [(Ellipsis, 42), (Ellipsis, .42, '.42')]

        Python has a perfectly good double-precision float type, but
        because EDN equality is different, it would coaus collections
        to fail to round-trip in some cases.
        >>> next(SimpleEDN.reads('#{1 1.0}'))
        {1}

        LiteralEDN can handle this without loss.
        >>> next(LiteralEDN.reads('#{1 1.0 1N 1M}'))
        (Ellipsis, {1, (Ellipsis, 1.0)})
        """
        return ..., float(v[:-1]), v
    def symbol(self, v):
        return bytes(v, encoding='ascii')
    keyword = symbol
    def bool(self, v):
        """Parses an EDN boolean as an Ellipsis group containing a bool.

        Python has a perfectly good bool type, but because EDN equality
        is different, it would cause collections to fail to round-trip
        in some cases.

        True is a special case of 1 in Python, so the first value was
        overwritten.
        >>> next(SimpleEDN.reads('{1 2, true 3}'))
        {1: 3}

        LiteralEDN can handle this without loss.
        >>> next(LiteralEDN.reads('{1 2, true 3}'))
        (Ellipsis, {1: 2, (Ellipsis, True): 3})
        """
        return ..., {'true': True, 'false': False}[v]
    def char(self, v):
        return ..., super().char(v)
    def tag(self, tag, v):
        return ..., bytes(tag, encoding='ascii'), v

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