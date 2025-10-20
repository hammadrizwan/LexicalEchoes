"""
Token types for the lexical analyzer.
Defines all possible token types that can be recognized by the lexer.
"""

from enum import Enum, auto


class TokenType(Enum):
    """Enumeration of all token types."""
    
    # Keywords
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    RETURN = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    TRUE = auto()
    FALSE = auto()
    VOID = auto()
    
    # Identifiers and Literals
    IDENTIFIER = auto()
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    
    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    MULTIPLY = auto()      # *
    DIVIDE = auto()        # /
    MODULO = auto()        # %
    ASSIGN = auto()        # =
    EQUAL = auto()         # ==
    NOT_EQUAL = auto()     # !=
    LESS_THAN = auto()     # <
    GREATER_THAN = auto()  # >
    LESS_EQUAL = auto()    # <=
    GREATER_EQUAL = auto() # >=
    AND = auto()           # &&
    OR = auto()            # ||
    NOT = auto()           # !
    
    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    SEMICOLON = auto()     # ;
    COMMA = auto()         # ,
    DOT = auto()           # .
    
    # Special
    EOF = auto()
    NEWLINE = auto()
    COMMENT = auto()


class Token:
    """Represents a single token with its type, value, and position."""
    
    def __init__(self, token_type, value, line, column):
        """
        Initialize a token.
        
        Args:
            token_type (TokenType): The type of the token
            value (str): The actual value/lexeme of the token
            line (int): Line number where the token appears
            column (int): Column number where the token starts
        """
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        """String representation of the token."""
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
    
    def __str__(self):
        """User-friendly string representation."""
        return f"{self.type.name}('{self.value}')"


# Keywords mapping
KEYWORDS = {
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'while': TokenType.WHILE,
    'for': TokenType.FOR,
    'return': TokenType.RETURN,
    'int': TokenType.INT,
    'float': TokenType.FLOAT,
    'string': TokenType.STRING,
    'bool': TokenType.BOOL,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'void': TokenType.VOID,
}
