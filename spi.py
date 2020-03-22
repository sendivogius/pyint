""" SPI - Simple Pascal Interpreter """
import argparse
import sys
from enum import Enum


class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUNT = 'Identifier not found'
    DUPLICATE_ID = 'Duplicate id found'


class Error(Exception):
    def __init__(self, error_code=None, token=None, message=None):
        self.error_code = error_code
        self.token = token
        self.message = f'{self.__class__.__name__}: {message}'


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class SemanticError(Error):
    pass


###############################################################################
#                                                                             #
#  SYMBOL PARSER                                                                      #
#                                                                             #
###############################################################################


class Symbol:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type


class BuiltinTypeSymbol(Symbol):

    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__}(name={self.name})>'


class VarSymbol(Symbol):
    def __init__(self, name, type):
        super().__init__(name, type)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{self.__class__.__name__}(name={self.name}, type={self.type})>'


class ProcedureSymbol(Symbol):

    def __init__(self, name, params=None):
        super().__init__(name)
        self.params = params or []

    def __str__(self):
        return '<{class_name}(name={name}, parameters={params})>'.format(
            class_name=self.__class__.__name__,
            name=self.name,
            params=self.params,
        )

    __repr__ = __str__


class ScopedSymbolTable:
    def __init__(self, name, level, enclosing_scope=None):
        self._symbols = dict()
        self.scope_level = level
        self.scope_name = name
        self.enclosing_scope = enclosing_scope

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def _init_builtins(self):
        self.define(BuiltinTypeSymbol('INTEGER'))
        self.define(BuiltinTypeSymbol('REAL'))

    def __str__(self):
        h1 = 'SCOPE (SCOPED SYMBOL TABLE)'
        lines = ['\n', h1, '=' * len(h1)]
        for header_name, header_value in (
                ('Scope name', self.scope_name),
                ('Scope level', self.scope_level),
                ('Enclosing scope',
                 self.enclosing_scope.scope_name if self.enclosing_scope else None
                 )
        ):
            lines.append('%-15s: %s' % (header_name, header_value))
        h2 = 'Scope (Scoped symbol table) contents'
        lines.extend([h2, '-' * len(h2)])
        lines.extend(
            ('%7s: %r' % (key, value))
            for key, value in self._symbols.items()
        )
        lines.append('\n')
        s = '\n'.join(lines)
        return s

    __repr__ = __str__

    def lookup(self, name, current_scope_only=False):
        self.log('Lookup: %s. (Scope name: %s)' % (name, self.scope_name))
        symbol = self._symbols.get(name)
        if symbol is not None:
            return symbol

        if current_scope_only:
            return None

        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)

    def define(self, symbol):
        self.log(f'Define: {symbol.name}')
        self._symbols[symbol.name] = symbol


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
class TokenType(Enum):
    # single-character token types
    PLUS = '+'
    MINUS = '-'
    MUL = '*'
    FLOAT_DIV = '/'
    LPAREN = '('
    RPAREN = ')'
    SEMI = ';'
    DOT = '.'
    COLON = ':'
    COMMA = ','
    # block of reserved words
    PROGRAM = 'PROGRAM'  # marks the beginning of the block
    INTEGER = 'INTEGER'
    REAL = 'REAL'
    INTEGER_DIV = 'DIV'
    VAR = 'VAR'
    PROCEDURE = 'PROCEDURE'
    BEGIN = 'BEGIN'
    END = 'END'  # marks the end of the block
    # misc
    ID = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST = 'REAL_CONST'
    ASSIGN = ':='
    EOF = 'EOF'


def _build_reserved_keywowrds():
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.PROGRAM)
    end_index = tt_list.index(TokenType.END)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index + 1]
    }
    return reserved_keywords


class Token(object):
    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.lineno = 1
        self.column = 1

    def error(self):
        s = f'Lexer error on {self.current_char} line: {self.lineno} column: {self.column}'
        raise LexerError(s)

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()  # closing bracket

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    RESERVED_KEYWORDS = _build_reserved_keywowrds()

    @staticmethod
    def can_be_id_start(c):
        return c.isalnum() or c == '_'

    def _id(self):
        """Handle identifiers and reserved keywords"""
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        result = ''
        while self.current_char is not None and self.can_be_id_start(self.current_char):
            result += self.current_char
            self.advance()

        token_type = self.RESERVED_KEYWORDS.get(result.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = result
        else:
            # reserved keyword
            token.type = token_type
            token.value = result.upper()

        return token

        token = self.RESERVED_KEYWORDS.get(result.upper(), Token(TokenType.ID, result))
        return token

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char != '.':
            return Token(TokenType.INTEGER_CONST, int(result), self.lineno, self.column)

        result += self.current_char
        self.advance()
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return Token(TokenType.REAL_CONST, float(result), self.lineno, self.column)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                self.skip_comment()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.can_be_id_start(self.current_char):
                return self._id()

            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.ASSIGN, ':=', self.lineno, self.column)

            try:
                # get enum member by value, e.g.
                # TokenType(';') --> TokenType.SEMI
                token_type = TokenType(self.current_char)
            except ValueError:
                # no enum member with value equal to self.current_char
                self.error()
            else:
                # create a token with a single-character lexeme as its value
                token = Token(
                    type=token_type,
                    value=token_type.value,  # e.g. ';', '.', etc
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

        return Token(TokenType.EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    def __init__(self):
        self.children = []


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Program(AST):
    def __init__(self, name, block):
        self.name = name
        self.block = block


class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement


class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node


class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class ProcedureDecl(AST):
    def __init__(self, proc_name, params, block_node):
        self.proc_name = proc_name
        self.params = params
        self.block_node = block_node


class NoOp(AST):
    pass


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Param(AST):

    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}'
        )

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(ErrorCode.UNEXPECTED_TOKEN, self.current_token)

    def program(self):
        """program : compound_statement DOT"""
        self.eat(TokenType.PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(TokenType.SEMI)
        block = self.block()
        node = Program(prog_name, block)
        self.eat(TokenType.DOT)
        return node

    def compound_statement(self):
        """
        compound_statement: BEGIN statement_list END
        """
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        """
        statement_list : statement
                       | statement SEMI statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        """
        statement : compound_statement
                  | assignment_statement
                  | empty
        """
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == TokenType.ID:
            node = self.assignment_statement()
        else:
            node = self.empty()
        return node

    def assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.variable()
        token = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        """
        variable : ID
        """
        node = Var(self.current_token)
        self.eat(TokenType.ID)
        return node

    def empty(self):
        """An empty production"""
        return NoOp()

    def factor(self):
        """factor : INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        if token.type in (TokenType.PLUS, TokenType.MINUS):
            self.eat(TokenType.token.type)
            node = UnaryOp(token, self.expr())
            return node

        if token.type in (TokenType.INTEGER_CONST, TokenType.REAL_CONST):
            self.eat(token.type)
            return Num(token)

        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        node = self.variable()
        return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (TokenType.MUL, TokenType.INTEGER_DIV, TokenType.FLOAT_DIV):
            token = self.current_token
            if token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif token.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)
            elif token.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """
        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : INTEGER | LPAREN expr RPAREN
        """
        node = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

        return node

    def block(self):
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        """
        declarations : (VAR (variable_declaration SEMI)+)? procedure_declaration*
        """
        declarations = []

        if self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                var_decl = self.variable_declaration()
                declarations.extend(var_decl)
                self.eat(TokenType.SEMI)

        while self.current_token.type == TokenType.PROCEDURE:
            proc_decl = self.procedure_declaration()
            declarations.append(proc_decl)

        return declarations

    def procedure_declaration(self):
        """procedure_declaration :
             PROCEDURE ID (LPAREN formal_parameter_list RPAREN)? SEMI block SEMI
        """
        self.eat(TokenType.PROCEDURE)
        proc_name = self.current_token.value
        self.eat(TokenType.ID)
        params = []

        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            params = self.formal_parameter_list()
            self.eat(TokenType.RPAREN)

        self.eat(TokenType.SEMI)
        block_node = self.block()
        proc_decl = ProcedureDecl(proc_name, params, block_node)
        self.eat(TokenType.SEMI)
        return proc_decl

    def formal_parameter_list(self):
        if not self.current_token.type == TokenType.ID:
            return []
        param_nodes = self.formal_parameters()
        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            param_nodes.extend(self.formal_parameters())

        return param_nodes

    def formal_parameters(self):
        param_tokens = [self.current_token]
        self.eat(TokenType.ID)
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            param_tokens.append(self.current_token)
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)
        type_node = self.type_spec()

        param_nodes = [Param(Var(param_token), type_node) for param_token in param_tokens]
        return param_nodes

    def variable_declaration(self):
        var_nodes = [Var(self.current_token)]
        self.eat(TokenType.ID)
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(TokenType.ID)
        self.eat(TokenType.COLON)

        type_node = self.type_spec()
        var_declarations = [VarDecl(vn, type_node) for vn in var_nodes]
        return var_declarations

    def type_spec(self):
        token = self.current_token
        if token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
        else:
            self.eat(TokenType.REAL)
        return Type(token)


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree

    def visit_BinOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == TokenType.FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.expr)
        elif node.op.type == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_NoOp(self, node):
        pass

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass

    def visit_Type(self, node):
        pass

    def visit_ProcedureDecl(self, node):
        pass

    GLOBAL_SCOPE = dict()

    def visit_Assign(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)
        return val

    def visit_Num(self, node):
        return node.value

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)


class RPN(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_BinOp(self, node):
        return self.visit(node.left) + ' ' + self.visit(node.right) + ' ' + node.op.value

    def visit_Num(self, node):
        return str(node.value)

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)


class LispNotation(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_BinOp(self, node):
        return '(' + node.op.value + ' ' + self.visit(node.left) + ' ' + self.visit(node.right) + ')'

    def visit_Num(self, node):
        return str(node.value)

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)


class SemanticAnalyzer(NodeVisitor):
    def __init__(self):
        self.current_scope: ScopedSymbolTable = None

    def error(self, error_code, token):
        raise SemanticError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_Num(self, node):
        pass

    def visit_UnaryOp(self, node):
        pass

    def visit_Program(self, node):
        print("ENTER scope global")
        global_scope = ScopedSymbolTable('global', 1, self.current_scope)
        global_scope._init_builtins()
        self.current_scope = global_scope
        self.visit(node.block)
        self.current_scope = self.current_scope.enclosing_scope
        print(global_scope)
        print("LEAVE scope global")

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_Num(self, node):
        pass

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_NoOp(self, node):
        pass

    def visit_VarDecl(self, node):
        type_name = node.type_node.value
        type_symbol = self.current_scope.lookup(type_name)

        var_name = node.var_node.value
        var_symbol = VarSymbol(var_name, type_symbol)

        if self.current_scope.lookup(var_name, current_scope_only=True):
            self.error(error_code=ErrorCode.DUPLICATE_ID, token=node.var_node.token)

        self.current_scope.define(var_symbol)

    def visit_Assign(self, node):
        self.visit(node.right)
        self.visit(node.left)

    def visit_Var(self, node):
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)

        if var_symbol is None:
            self.error(error_code=ErrorCode.ID_NOT_FOUNT, token=node.token)

    def visit_ProcedureDecl(self, node):
        proc_name = node.proc_name
        proc_symbol = ProcedureSymbol(proc_name)
        self.current_scope.define(proc_symbol)

        print(f'ENTER scope: {proc_name}')
        proc_scope = ScopedSymbolTable(proc_name, self.current_scope.scope_level + 1, self.current_scope)
        self.current_scope = proc_scope

        for param in node.params:
            param_type = self.current_scope.lookup(param.type_node.value)
            param_name = param.var_node.value
            var_symbol = VarSymbol(param_name, param_type)
            self.current_scope.define(var_symbol)
            proc_symbol.params.append(var_symbol)
        self.visit(node.block_node)
        print(proc_scope)
        self.current_scope = self.current_scope.enclosing_scope
        print(f'LEAVE scope: {proc_name}')


def main():
    parser = argparse.ArgumentParser(
        description='SPI - Simple Pascal Interpreter'
    )
    parser.add_argument('inputfile', help='Pascal source file')
    parser.add_argument(
        '--scope',
        help='Print scope information',
        action='store_true',
    )
    args = parser.parse_args()
    global _SHOULD_LOG_SCOPE
    _SHOULD_LOG_SCOPE = args.scope

    text = open(args.inputfile, 'r').read()

    lexer = Lexer(text)
    try:
        parser = Parser(lexer)
        tree = parser.parse()
    except (LexerError, ParserError) as e:
        print(e.message)
        sys.exit(1)

    semantic_analyzer = SemanticAnalyzer()
    try:
        semantic_analyzer.visit(tree)
    except SemanticError as e:
        print(e.message)
        sys.exit(1)

    interpreter = Interpreter(tree)
    interpreter.interpret()


if __name__ == '__main__':
    main()
