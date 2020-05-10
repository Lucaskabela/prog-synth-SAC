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

-- See https://github.com/yeoedward/Robust-Fill/
'''
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from functools import reduce
from string import printable, whitespace
import re

#########################################
######## ABSTRACT CLASS / DESIGN# #######
######################################### 
class DSL(ABC):

    @abstractmethod
    def eval(self, value):
        '''
        Defines the semantics of how to evaluate value
        '''
        raise NotImplementedError

    @abstractmethod
    def to_string(self, indent, tab):
        '''
        convert the program to string representation, with indentation for readability
        '''
        raise NotImplementedError

    def __repr__(self):
        return self.to_string(indent=0, tab=4)

    @abstractmethod
    def to_tokens(self, op_token_table):
        '''
        Convert DSL to token version from a table
        '''
        raise NotImplementedError

class Program(DSL):
    pass


class Expression(DSL):
    pass


class Substring(Expression):
    pass


class Nesting(Expression):
    pass

#########################################
######## LITERAL PARAMETER VALUES #######
######################################### 
class CASE(Enum):
    PROPER = 1
    ALL_CAPS = 2
    LOWER = 3

# Positions are [-100, 100] in strings, limited due to finite output
POSITION = list(range(-100, 101))

# Index for occurences are -5 to 5, with negative values from the end of the string
INDEX = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

CHARACTER = printable

DELIMITER = '&,.?!@()[]%{}/:;$#' + whitespace

# Boundary for GetSpan
class BOUNDARY(Enum):
    START = 1
    END = 2

class Type(Enum):
    NUMBER = 1
    WORD = 2
    ALPHANUM = 3
    ALL_CAPS = 4
    PROP_CASE = 5
    LOWER = 6
    DIGIT = 7
    CHAR = 8

#########################################
######## DEFINE FUNCTIONS IN DSL ######## 
#########################################

class Concat(Program):
    '''
    Concat is the top level operator.  Concatenates together the subexpressions
    '''
    def __init__(self, *args):
        self.expressions = args

    def eval(self, value):
        return ''.join([e.eval(value) for e in self.expressions])

    def to_string(self, indent, tab):
        ind = indent + tab
        sub_exps = [e.to_string(indent=ind, tab=tab) for e in self.expressions]
        return op_to_string('Concat', sub_exps, indent, recursive=True)

    def to_tokens(self, op_token_table):
        sub_tkns = [e.to_tokens(op_token_table) for e in self.expressions]
        return reduce(
                lambda a, b: a + [op_token_table[self.__class__]] + b, 
                sub_tkns) + [op_token_table['EOS']]


class Compose(Nesting):
    '''
    Composes two operators:  A nesting, and another nesting or substring to allow for more rich
    expressions.
    '''
    def __init__(self, nesting, nesting_or_substring):
        self.nesting = nesting
        self.nesting_or_substring = nesting_or_substring

    def eval(self, value):
        return self.nesting.eval(self.nesting_or_substring.eval(value))

    def to_string(self, indent, tab):
        ind = indent + tab
        nesting = self.nesting.to_string(indent=ind, tab=tab)
        nesting_or_substring = self.nesting_or_substring.to_string(
                indent=ind, tab=tab)

        return op_to_string('Compose', [nesting, nesting_or_substring],
                indent, recursive=True)

    def to_tokens(self, op_token_table):
        sub_exp = self.nesting.to_tokens(op_token_table) + self.nesting_or_substring.to_tokens(op_token_table)
        return [op_token_table[self.__class__]] + sub_exp


class ConstStr(Expression):
    '''
    Returns the constant string (which is limited to a character in this implementaion)
        to form words, simply concat multiple constant strings.
    '''
    def __init__(self, string):
        self.string = string

    def eval(self, in_str):
        return self.string

    def to_string(self, indent, tab):
        return op_to_string('ConstStr', [self.string], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[self.__class__], op_token_table[self.string]]


class SubStr(Substring):
    '''
    Returns the substring starting at pos1 and ending at pos2, both of which are ints.
        negative values go to end of string.
    '''
    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2

    @staticmethod
    def _substr_index(pos, in_str):
        # DSL index starts at one
        if pos > 0:
            return pos - 1

        # Prevent underflow
        if abs(pos) > len(in_str):
            return 0

        return pos

    def eval(self, in_str):
        p1 = SubStr._substr_index(self.pos1, in_str)
        p2 = SubStr._substr_index(self.pos2, in_str)

        # Edge case: When p2 == -1, incrementing by one doesn't
        # make it inclusive. Instead, an empty string is always returned.
        if p2 == -1:
            return in_str[p1:]

        return in_str[p1:p2+1]

    def to_string(self, indent, tab):
        return op_to_string('SubStr', [self.pos1, self.pos2], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[self.__class__], op_token_table[self.pos1],
                op_token_table[self.pos2]]


class GetSpan(Substring):
    '''
    GetSpan operation will get substring starting at the index1 occurence of regex1, and include or
    not based on bound1.  It ends at the index2 occurence of regex2, including based on bound2
    '''
    def __init__(self, dsl_regex1, index1, bound1, dsl_regex2, index2, bound2):
        self.dsl_regex1 = dsl_regex1
        self.index1 = index1
        self.bound1 = bound1
        self.dsl_regex2 = dsl_regex2
        self.index2 = index2
        self.bound2 = bound2

    # By convention, we always prefix the DSL regex with 'dsl_' as a way to
    # distinguish it with regular regexes.
    @staticmethod
    def _span_index(dsl_regex, index, bound, in_str):
        matches = match_dsl_regex(dsl_regex, in_str)
        # Positive indices start at 1, so we need to substract 1
        index = index if index < 0 else index - 1

        if index >= len(matches):
            return len(matches) - 1

        if index < -len(matches):
            return 0

        span = matches[index]
        return span[0] if bound == BOUNDARY.START else span[1]

    def eval(self, in_str):
        p1 = GetSpan._span_index(self.dsl_regex1, self.index1, self.bound1,
                in_str)
        p2 = GetSpan._span_index(self.dsl_regex2, self.index2, self.bound2,
                in_str)
        return in_str[p1:p2]

    def to_string(self, indent, tab):
        return op_to_string('GetSpan',[self.dsl_regex1, self.index1,
                self.bound1, self.dsl_regex2, self.index2, self.bound2],
                indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[elem]
            for elem in [self.__class__, self.dsl_regex1, self.index1,
                self.bound1, self.dsl_regex2, self.index2, self.bound2]]


class GetToken(Nesting):
    '''
    Gets the index occurence of a token type type_
    '''
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index

    def eval(self, in_str):
        matches = match_type(self.type_, in_str)
        i = self.index
        if self.index > 0:
            # Positive indices start at 1
            i -= 1
        return matches[i]

    def to_string(self, indent, tab):
        return op_to_string('GetToken', [self.type_, self.index], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_, self.index)]]


class ToCase(Nesting):
    '''
    Converts the to one of the enumerated cases
    '''
    def __init__(self, case):
        self.case = case

    def eval(self, in_str):
        if self.case == CASE.PROPER:
            return in_str.capitalize()
        elif self.case == CASE.ALL_CAPS:
            return in_str.upper()
        elif self.case == CASE.LOWER:
            return in_str.lower()
        else:
            raise ValueError('Invalid case: {}'.format(self.case))

    def to_string(self, indent, tab):
        return op_to_string('ToCase', [self.case], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.case)]]


class Replace(Nesting):
    '''
    Replaces all occurences of delim1 with delim2
    '''
    def __init__(self, delim1, delim2):
        self.delim1 = delim1
        self.delim2 = delim2

    def eval(self, in_str):
        return in_str.replace(self.delim1, self.delim2)

    def to_string(self, indent, tab):
        return op_to_string('Replace', [self.delim1, self.delim2], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.delim1, self.delim2)]]


class Trim(Nesting):
    '''
    Trims whitespace and garbage from the string
    '''
    def eval(self, in_str):
        return in_str.strip()

    def to_string(self, indent, tab):
        return op_to_string('Trim', [], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[self.__class__]]


class GetUpto(Nesting):
    '''
    Get all characters up to the first occurence of the regex
    '''
    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex

    def eval(self, in_str):
        matches = match_dsl_regex(self.dsl_regex, in_str)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return in_str[:first[1]]

    def to_string(self, indent, tab):
        return op_to_string('GetUpto', [self.dsl_regex], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.dsl_regex)]]


class GetFrom(Nesting):
    '''
    Gets all characters after the occurence of regex
    '''
    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex

    def eval(self, in_str):
        matches = match_dsl_regex(self.dsl_regex, in_str)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return in_str[first[1]:]

    def to_string(self, indent, tab):
        return op_to_string('GetFrom', [self.dsl_regex], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.dsl_regex)]]


class GetFirst(Nesting):
    '''
    Gets the first token of type_ starting from index.
    '''
    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index

    def eval(self, in_str):
        matches = match_type(self.type_, in_str)

        if self.index < 0:
            raise IndexError

        return ''.join(matches[:self.index])

    def to_string(self, indent, tab):
        return op_to_string('GetFirst', [self.type_, self.index], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_, self.index)]]


class GetAll(Nesting):
    '''
    Gets all tokens of type_
    '''
    def __init__(self, type_):
        self.type_ = type_

    def eval(self, in_str):
        return ' '.join(match_type(self.type_, in_str))

    def to_string(self, indent, tab):
        return op_to_string('GetAll', [self.type_], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_)]]



def match_type(type_, in_str):
    regex = regex_for_type(type_)
    return re.findall(regex, in_str)


def match_dsl_regex(dsl_regex, in_str):
    if isinstance(dsl_regex, Type):
        regex = regex_for_type(dsl_regex)
    else:
        assert len(dsl_regex) == 1 and dsl_regex in DELIMITER
        regex = '[' + re.escape(dsl_regex) + ']'

    return [
        match.span()
        for match in re.finditer(regex, in_str)
    ]


def regex_for_type(type_):
    if type_ == Type.NUMBER:
        return '[0-9]+'

    if type_ == Type.WORD:
        return '[A-Za-z]+'

    if type_ == Type.ALPHANUM:
        return '[A-Za-z0-9]+'

    if type_ == Type.ALL_CAPS:
        return '[A-Z]+'

    if type_ == Type.PROP_CASE:
        return '[A-Z][a-z]+'

    if type_ == Type.LOWER:
        return '[a-z]+'

    if type_ == Type.DIGIT:
        return '[0-9]'
        
    if type_ == Type.CHAR:
        return '[A-Za-z0-9]'

    raise ValueError('Unsupported type: {}'.format(type_))


def op_to_string(name, raw_args, indent, recursive=False):
    indent_str = ' ' * indent

    if recursive:
        args_str = ',\n'.join(raw_args)
        return '{indent_str}{name}(\n{args_str}\n{indent_str})'.format(
            indent_str=indent_str,
            name=name,
            args_str=args_str,
        )

    args = [repr(a) for a in raw_args]
    args_str = ', '.join(args)
    return '{indent_str}{name}({args_str})'.format(
        indent_str=indent_str,
        name=name,
        args_str=args_str,
    )


#########################################
########### OP TOKEN TABLES  ############ 
#########################################

# Special token for end-of-sequence
EOS = 'EOS'
TokenTables = namedtuple(
    'TokenTables',
    ['token_op_table', 'op_token_table', 'string_token_table'],
)


def tokenize_string(string, string_token_table):
    '''
    Converts string into tokens, using the string_token_table 
    '''
    return [string_token_table[char] for char in string]

def stringify_tokens(tokens, string_token_table):
    '''
    Converts a list of tokens into strings based on the string_token_table (flips the k, v in this table)
    '''
    token_string_table = {token: char for token, char in enumerate(string_token_table)}
    return ''.join([token_string_table[tok] for tok in tokens])


def build_token_tables():
    '''
    Constructs the token_op, op_token, and string_token tables and returns these in one named tuple
    '''
    token_op_table = [EOS, Concat, Compose, ConstStr, SubStr,
            GetSpan, Trim]

    # Nesting operators and their args get "compacted" into "primitive" tokens
    for type_ in Type:
        for index in INDEX:
            token_op_table.append((GetToken, type_, index))

    for case in CASE:
        token_op_table.append((ToCase, case))

    for delim1 in DELIMITER:
        for delim2 in DELIMITER:
            token_op_table.append((Replace, delim1, delim2))

    for dsl_regex in list(Type) + list(DELIMITER):
        token_op_table.append((GetUpto, dsl_regex))

    for dsl_regex in list(Type) + list(DELIMITER):
        token_op_table.append((GetFrom, dsl_regex))

    for type_ in Type:
        for index in INDEX:
            token_op_table.append((GetFirst, type_, index))

    for type_ in Type:
        token_op_table.append((GetAll, type_))

    # Primitive types
    for type_ in Type:
        token_op_table.append(type_)

    for case in CASE:
        token_op_table.append(case)

    for boundary in BOUNDARY:
        token_op_table.append(boundary)

    # Covers index too
    for position in POSITION:
        token_op_table.append(position)

    # This covers DELIMITER
    for character in CHARACTER:
        token_op_table.append(character)

    token_op_table = {token: op for token, op in enumerate(token_op_table)}

    op_token_table = {op: token for token, op in token_op_table.items()}

    assert len(token_op_table) == len(op_token_table)

    string_token_table = {char: token for token, char in enumerate(CHARACTER)}

    return TokenTables(token_op_table=token_op_table, op_token_table=op_token_table,
            string_token_table=string_token_table)