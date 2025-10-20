"""
Lexical Analyzer (Lexer/Scanner) for tokenizing source code.
Converts raw source code into a stream of tokens.
"""

from token_types import Token, TokenType, KEYWORDS


class LexicalError(Exception):
    """Exception raised for lexical analysis errors."""
    
    def __init__(self, message, line, column):
        super().__init__(f"Lexical Error at {line}:{column} - {message}")
        self.line = line
        self.column = column


class Lexer:
    """Lexical analyzer that tokenizes source code."""
    
    def __init__(self, source_code):
        """
        Initialize the lexer with source code.
        
        Args:
            source_code (str): The source code to tokenize
        """
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if source_code else None
    
    def advance(self):
        """Move to the next character in the source code."""
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        self.position += 1
        if self.position >= len(self.source):
            self.current_char = None
        else:
            self.current_char = self.source[self.position]
    
    def peek(self, offset=1):
        """
        Look ahead at the next character without consuming it.
        
        Args:
            offset (int): How many characters to look ahead
            
        Returns:
            str or None: The character at the offset position
        """
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        """Skip whitespace characters except newlines."""
        while self.current_char and self.current_char in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments (// ...) and multi-line comments (/* ... */)."""
        if self.current_char == '/' and self.peek() == '/':
            # Single-line comment
            while self.current_char and self.current_char != '\n':
                self.advance()
        elif self.current_char == '/' and self.peek() == '*':
            # Multi-line comment
            self.advance()  # Skip /
            self.advance()  # Skip *
            while self.current_char:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()  # Skip *
                    self.advance()  # Skip /
                    break
                self.advance()
    
    def read_number(self):
        """
        Read a numeric literal (integer or float).
        
        Returns:
            Token: INTEGER_LITERAL or FLOAT_LITERAL token
        """
        start_line = self.line
        start_column = self.column
        num_str = ''
        is_float = False
        
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if is_float:
                    raise LexicalError("Invalid number format", self.line, self.column)
                is_float = True
            num_str += self.current_char
            self.advance()
        
        token_type = TokenType.FLOAT_LITERAL if is_float else TokenType.INTEGER_LITERAL
        return Token(token_type, num_str, start_line, start_column)
    
    def read_string(self):
        """
        Read a string literal enclosed in double quotes.
        
        Returns:
            Token: STRING_LITERAL token
        """
        start_line = self.line
        start_column = self.column
        string_value = ''
        
        self.advance()  # Skip opening quote
        
        while self.current_char and self.current_char != '"':
            if self.current_char == '\\':
                self.advance()
                if self.current_char in 'ntr"\\':
                    escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}
                    string_value += escape_chars.get(self.current_char, self.current_char)
                    self.advance()
                else:
                    string_value += self.current_char
                    self.advance()
            else:
                string_value += self.current_char
                self.advance()
        
        if self.current_char != '"':
            raise LexicalError("Unterminated string literal", start_line, start_column)
        
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING_LITERAL, string_value, start_line, start_column)
    
    def read_identifier(self):
        """
        Read an identifier or keyword.
        
        Returns:
            Token: IDENTIFIER or keyword token
        """
        start_line = self.line
        start_column = self.column
        identifier = ''
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            identifier += self.current_char
            self.advance()
        
        # Check if it's a keyword
        token_type = KEYWORDS.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, start_line, start_column)
    
    def get_next_token(self):
        """
        Get the next token from the source code.
        
        Returns:
            Token: The next token in the source code
        """
        while self.current_char:
            # Skip whitespace
            if self.current_char in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Handle newlines
            if self.current_char == '\n':
                token = Token(TokenType.NEWLINE, '\\n', self.line, self.column)
                self.advance()
                return token
            
            # Handle comments
            if self.current_char == '/' and self.peek() in '/*':
                self.skip_comment()
                continue
            
            # Numbers
            if self.current_char.isdigit():
                return self.read_number()
            
            # Strings
            if self.current_char == '"':
                return self.read_string()
            
            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()
            
            # Operators and delimiters
            start_line = self.line
            start_column = self.column
            char = self.current_char
            
            # Two-character operators
            if char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.EQUAL, '==', start_line, start_column)
            elif char == '!' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.NOT_EQUAL, '!=', start_line, start_column)
            elif char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.LESS_EQUAL, '<=', start_line, start_column)
            elif char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.GREATER_EQUAL, '>=', start_line, start_column)
            elif char == '&' and self.peek() == '&':
                self.advance()
                self.advance()
                return Token(TokenType.AND, '&&', start_line, start_column)
            elif char == '|' and self.peek() == '|':
                self.advance()
                self.advance()
                return Token(TokenType.OR, '||', start_line, start_column)
            
            # Single-character operators and delimiters
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS_THAN,
                '>': TokenType.GREATER_THAN,
                '!': TokenType.NOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
            }
            
            if char in single_char_tokens:
                self.advance()
                return Token(single_char_tokens[char], char, start_line, start_column)
            
            # Unknown character
            raise LexicalError(f"Unexpected character '{char}'", start_line, start_column)
        
        # End of file
        return Token(TokenType.EOF, '', self.line, self.column)
    
    def tokenize(self):
        """
        Tokenize the entire source code.
        
        Returns:
            list[Token]: List of all tokens in the source code
        """
        tokens = []
        while True:
            token = self.get_next_token()
            if token.type != TokenType.NEWLINE:  # Skip newline tokens in the list
                tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens
