# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
INTEGER, PLUS, MINUS, WS, EOF = 'INTEGER', 'PLUS', 'MINUS', "WS", 'EOF'


class Token(object):
    def __init__(self, type, value):
        self.type = type
        # token value: 0, 1, 2. 3, 4, 5, 6, 7, 8, 9, '+', or None
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS '+')
        """
        return f'Token({self.type}, {self.value})'

    def __repr__(self):
        return self.__str__()


class Interpreter(object):
    def __init__(self, text):
        # client string input, e.g. "3+5"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        # current token instance
        self.current_token = None

    def error(self):
        raise Exception('Error parsing input')

    def get_next_raw_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        text = self.text

        # is self.pos index past the end of the self.text ?
        # if so, then return EOF token because there is no more
        # input left to convert into tokens
        if self.pos > len(text) - 1:
            return Token(EOF, None)

        # get a character at the position self.pos and decide
        # what token to create based on the single character
        current_char = text[self.pos]

        if current_char.isdigit():
            start_pos = self.pos
            while current_char.isdigit():
                self.pos += 1
                current_char = text[self.pos] if self.pos < len(text) else ''
            return Token(INTEGER, int(self.text[start_pos:self.pos]))

        if current_char == '+':
            self.pos += 1
            return Token(PLUS, current_char)

        if current_char == '-':
            self.pos += 1
            return Token(MINUS, current_char)

        if current_char.isspace():
            self.pos += 1
            return Token(WS, current_char)

        self.error()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error()

    def get_next_token(self):
        token = self.get_next_raw_token()
        while token.type == WS:
            token = self.get_next_raw_token()
        return token

    def expr(self):
        """expr -> INTEGER PLUS INTEGER"""
        # set current token to the first token taken from the input
        self.current_token = self.get_next_token()

        # we expect the current token to be a single-digit integer
        left = self.current_token
        self.eat(INTEGER)

        # we expect the current token to be a '+' token
        op = self.current_token
        self.eat(PLUS if op.type == PLUS else MINUS)

        right = self.current_token
        self.eat(INTEGER)
        if op.type == PLUS:
            result = left.value + right.value
        else:
            result = left.value - right.value
        return result


def main():
    while True:
        try:
            text = input('calc> ')
        except EOFError:
            break
        if not text:
            continue
        interpreter = Interpreter(text)
        result = interpreter.expr()
        print(result)


if __name__ == '__main__':
    main()
