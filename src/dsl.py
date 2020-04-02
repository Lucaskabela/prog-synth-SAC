'''
This file defines the DSL for this project

- Concat (e1, e2, e3, ...) = Concat ( [e1] , [e2], [e3], ...)

	- ConstStr (c) = c

	- Substr (k1, k2) = v[p1...p2] where p is k if k > 0, else len(v) + p

	- GetSpan(r1, i1, y1, r2, i2, y2) = starting at i1th match of r1, y1 says start or end, end at i2th match of r2, y2 ""

	- _Nested Expressions_ [n1(n2)] = [n1]_v1_ where v1 = [n2]

	- GetToken(t, i) = ith match of t from beginning (end if i < 0)

	- GetUpto(r) = v[0...i] where i is the index at end of match of r

	- GetFrom(r) = v[j...len(v)] where j is end of last match of r 

	- GetFirst(t, i) = Concat first i matches of t

	- GetAll(t) = Concat all matches of t

	- ToCase(s) = upper or lower

	- Trim() = removes whitespace from around the string

	- Replace(delta1, delta2) = replaces regex delta1 with delta2
'''
from abc import ABC, abstractmethod
from enum import Enum
from string import printable, whitespace


class DSL(ABC):

    @abstractmethod
    def eval(self, value):
    	'''
    	Defines the semantics of how to evaluate value
    	'''
        raise NotImplementedError

	@abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def to_tokens(self, op_token_table):
    	'''
    	Convert DSL to token version from a table
    	'''
        raise NotImplementedError

class Program(DSL):
    pass


######## LITERAL PARAMETER VALUES ######## 
class CASE(Enum):
    PROPER = 1
    ALL_CAPS = 2
    LOWER = 3

# Positions are [-100, 100] in strings, limited due to finite output
POSITION = list(range(-100, 101))

# Index for occurences are -5 to 5, with -5 being from the end of the string
INDEX = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

CHARACTER = printable

DELIMITER = '&,.?!@()[]%{}/:;$#' + whitespace

# Boundary for GetSpan
class BOUNDARY(Enum):
    START = 1
    END = 2

#################################
######## DEFINE FUNCTIONS IN DSL ######## 

class Concat(Program):
	'''
	Concat is the top level command in this DSL and serves as the 
	driver for the program.  
	'''
    def __init__(self, *args):
        self.expressions = args

    def eval(self, value):
        return ''.join([
            e.eval(value)
            for e in self.expressions
        ])

    def to_string(self, indent, tab):
        sub_exps = [
            e.to_string(indent=indent+tab, tab=tab)
            for e in self.expressions
        ]
        return op_to_string('Concat', sub_exps, indent, recursive=True)

    def to_tokens(self, op_token_table):
        sub_tokens = [
            e.to_tokens(op_token_table)
            for e in self.expressions
        ]
        return reduce(
            lambda a, b: a + [op_token_table[self.__class__]] + b,
            sub_tokens,
        ) + [op_token_table['EOS']]
