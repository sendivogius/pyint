""" SPI - Simple Pascal Interpreter """

###############################################################################
#                                                                             #
#  SYMBOL PARSER                                                                      #
#                                                                             #
###############################################################################
from collections import OrderedDict


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

    def lookup(self, name):
        print('Lookup: %s. (Scope name: %s)' % (name, self.scope_name))
        symbol = self._symbols.get(name)
        if symbol is not None:
            return symbol

        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)

    def define(self, symbol):
        print(f'Define: {symbol.name}')
        self._symbols[symbol.name] = symbol


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
INTEGER, PLUS, MINUS, MUL, LPAREN, RPAREN, EOF, BEGIN, END, DOT, ASSIGN, SEMI, ID, PROGRAM, VAR, COLON, COMMA, REAL, INTEGER_CONST, REAL_CONST, INTEGER_DIV, FLOAT_DIV, PROCEDURE = (
    'INTEGER', 'PLUS', 'MINUS', 'MUL', '(', ')', 'EOF', 'BEGIN', 'END', 'DOT', 'ASSIGN', 'SEMI', 'ID', 'PROGRAM', 'VAR',
    'COLON', 'COMMA', 'REAL', 'INTEGER_CONST', 'REAL_CONST', 'INTEGER_DIV', 'FLOAT_DIV', 'PROCEDURE'
)


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
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

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

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

    RESERVED_KEYWORDS = {
        'BEGIN': Token(BEGIN, 'BEGIN'),
        'END': Token(END, 'END'),
        'DIV': Token(INTEGER_DIV, 'DIV'),
        'PROGRAM': Token(PROGRAM, 'PROGRAM'),
        'VAR': Token(VAR, 'VAR'),
        'REAL': Token(REAL, 'REAL'),
        'INTEGER': Token(INTEGER, 'INTEGER'),
        'PROCEDURE': Token(PROCEDURE, 'PROCEDURE')
    }

    @staticmethod
    def can_be_id_start(c):
        return c.isalnum() or c == '_'

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while self.current_char is not None and self.can_be_id_start(self.current_char):
            result += self.current_char
            self.advance()

        token = self.RESERVED_KEYWORDS.get(result.upper(), Token(ID, result))
        return token

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char != '.':
            return Token(INTEGER_CONST, int(result))

        result += self.current_char
        self.advance()
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return Token(REAL_CONST, float(result))

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
                return Token(ASSIGN, ':=')

            if self.current_char == ':':
                self.advance()
                return Token(COLON, ':')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(FLOAT_DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)


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

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        """program : compound_statement DOT"""
        self.eat(PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(SEMI)
        block = self.block()
        node = Program(prog_name, block)
        self.eat(DOT)
        return node

    def compound_statement(self):
        """
        compound_statement: BEGIN statement_list END
        """
        self.eat(BEGIN)
        nodes = self.statement_list()
        self.eat(END)

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

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results.append(self.statement())

        if self.current_token.type == ID:
            self.error()

        return results

    def statement(self):
        """
        statement : compound_statement
                  | assignment_statement
                  | empty
        """
        if self.current_token.type == BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == ID:
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
        self.eat(ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        """
        variable : ID
        """
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        """An empty production"""
        return NoOp()

    def factor(self):
        """factor : INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        if token.type in (PLUS, MINUS):
            self.eat(token.type)
            node = UnaryOp(token, self.expr())
            return node

        if token.type in (INTEGER_CONST, REAL_CONST):
            self.eat(token.type)
            return Num(token)

        if token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        node = self.variable()
        return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """
        expr   : term ((PLUS | MINUS) term)*
        term   : factor ((MUL | DIV) factor)*
        factor : INTEGER | LPAREN expr RPAREN
        """
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()

        return node

    def block(self):
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        declarations = []
        while True:
            if self.current_token.type == VAR:
                self.eat(VAR)
                while self.current_token.type == ID:
                    var_decl = self.variable_declaration()
                    declarations.extend(var_decl)
                    self.eat(SEMI)

            elif self.current_token.type == PROCEDURE:
                self.eat(PROCEDURE)
                proc_name = self.current_token.value
                self.eat(ID)
                params = []

                if self.current_token.type == LPAREN:
                    self.eat(LPAREN)
                    params = self.formal_parameter_list()
                    self.eat(RPAREN)

                self.eat(SEMI)
                block_node = self.block()
                proc_decl = ProcedureDecl(proc_name, params, block_node)
                declarations.append(proc_decl)
                self.eat(SEMI)
            else:
                break

        return declarations

    def formal_parameter_list(self):
        if not self.current_token.type == ID:
            return []
        param_nodes = self.formal_parameters()
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            param_nodes.extend(self.formal_parameters())

        return param_nodes

    def formal_parameters(self):
        param_tokens = [self.current_token]
        self.eat(ID)
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            param_tokens.append(self.current_token)
            self.eat(ID)

        self.eat(COLON)
        type_node = self.type_spec()

        param_nodes = [Param(Var(param_token), type_node) for param_token in param_tokens]
        return param_nodes

    def variable_declaration(self):
        var_nodes = [Var(self.current_token)]
        self.eat(ID)
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(ID)
        self.eat(COLON)

        type_node = self.type_spec()
        var_declarations = [VarDecl(vn, type_node) for vn in var_nodes]
        return var_declarations

    def type_spec(self):
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
        else:
            self.eat(REAL)
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
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.expr)
        elif node.op.type == MINUS:
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

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

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

        # if self.symtab.lookup(var_name) is not None:
        #     raise Exception(f'Error: Duplicate identifier found {var_name}')

        self.current_scope.define(var_symbol)

    def visit_Assign(self, node):
        self.visit(node.right)
        self.visit(node.left)

    def visit_Var(self, node):
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)

        if var_symbol is None:
            raise Exception(f'Error: Symbol (identifier) not found {var_name}')

    def visit_ProcedureDecl(self, node):
        proc_name = node.proc_name
        proc_symbol = ProcedureSymbol(proc_name)
        self.current_scope.define(proc_symbol)

        print(f'ENTER scope: {proc_name}')
        proc_scope = ScopedSymbolTable(proc_name, self.current_scope.scope_level+1, self.current_scope)
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
    text = """
program Main;
   var x, y: real;
   procedure Alpha(a : integer);
      var y : integer;
   begin
      x := a + x + y;
   end;
begin { Main }
end.  { Main }
"""
    lexer = Lexer(text)
    parser = Parser(lexer)
    tree = parser.parse()
    symtab_builder = SemanticAnalyzer()
    symtab_builder.visit(tree)
    print(symtab_builder.current_scope)
    interpreter = Interpreter(tree)
    interpreter.interpret()

    print('')
    print('Run-time GLOBAL_MEMORY contents:')
    for k, v in sorted(interpreter.GLOBAL_SCOPE.items()):
        print('{} = {}'.format(k, v))

    while True:
        try:
            try:
                text = raw_input('spi> ')
            except NameError:  # Python3
                text = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(result)


if __name__ == '__main__':
    main()
