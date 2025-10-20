"""
Unit tests for the lexer module.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lexer import Lexer, LexicalError
from token_types import TokenType


class TestLexer(unittest.TestCase):
    """Test cases for the Lexer class."""
    
    def test_keywords(self):
        """Test recognition of keywords."""
        source = "if else while for return int float string bool true false void"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.IF, TokenType.ELSE, TokenType.WHILE, TokenType.FOR,
            TokenType.RETURN, TokenType.INT, TokenType.FLOAT, TokenType.STRING,
            TokenType.BOOL, TokenType.TRUE, TokenType.FALSE, TokenType.VOID,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected in zip(tokens, expected_types):
            self.assertEqual(token.type, expected)
    
    def test_identifiers(self):
        """Test recognition of identifiers."""
        source = "x myVar _private count123"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(len(tokens), 5)  # 4 identifiers + EOF
        for i in range(4):
            self.assertEqual(tokens[i].type, TokenType.IDENTIFIER)
    
    def test_integer_literals(self):
        """Test recognition of integer literals."""
        source = "0 42 123 9999"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(len(tokens), 5)  # 4 integers + EOF
        for i in range(4):
            self.assertEqual(tokens[i].type, TokenType.INTEGER_LITERAL)
    
    def test_float_literals(self):
        """Test recognition of float literals."""
        source = "3.14 0.5 99.99"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(len(tokens), 4)  # 3 floats + EOF
        for i in range(3):
            self.assertEqual(tokens[i].type, TokenType.FLOAT_LITERAL)
    
    def test_string_literals(self):
        """Test recognition of string literals."""
        source = '"hello" "world" "test 123"'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(len(tokens), 4)  # 3 strings + EOF
        self.assertEqual(tokens[0].value, "hello")
        self.assertEqual(tokens[1].value, "world")
        self.assertEqual(tokens[2].value, "test 123")
    
    def test_operators(self):
        """Test recognition of operators."""
        source = "+ - * / % = == != < > <= >= && || !"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE,
            TokenType.MODULO, TokenType.ASSIGN, TokenType.EQUAL, TokenType.NOT_EQUAL,
            TokenType.LESS_THAN, TokenType.GREATER_THAN, TokenType.LESS_EQUAL,
            TokenType.GREATER_EQUAL, TokenType.AND, TokenType.OR, TokenType.NOT,
            TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected in zip(tokens, expected_types):
            self.assertEqual(token.type, expected)
    
    def test_delimiters(self):
        """Test recognition of delimiters."""
        source = "( ) { } [ ] ; , ."
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE, TokenType.RBRACE,
            TokenType.LBRACKET, TokenType.RBRACKET, TokenType.SEMICOLON,
            TokenType.COMMA, TokenType.DOT, TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected in zip(tokens, expected_types):
            self.assertEqual(token.type, expected)
    
    def test_comments(self):
        """Test that comments are skipped."""
        source = """
        int x = 5; // This is a comment
        /* This is a
           multi-line comment */
        int y = 10;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        # Should have: int, x, =, 5, ;, int, y, =, 10, ;, EOF
        self.assertEqual(len(tokens), 11)
        self.assertEqual(tokens[0].type, TokenType.INT)
        self.assertEqual(tokens[5].type, TokenType.INT)
    
    def test_variable_declaration(self):
        """Test tokenizing a variable declaration."""
        source = "int x = 10;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.INT, TokenType.IDENTIFIER, TokenType.ASSIGN,
            TokenType.INTEGER_LITERAL, TokenType.SEMICOLON, TokenType.EOF
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected in zip(tokens, expected_types):
            self.assertEqual(token.type, expected)
    
    def test_line_and_column_tracking(self):
        """Test that line and column numbers are tracked correctly."""
        source = "int x\nfloat y"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(tokens[0].line, 1)  # int
        self.assertEqual(tokens[1].line, 1)  # x
        self.assertEqual(tokens[2].line, 2)  # float
        self.assertEqual(tokens[3].line, 2)  # y
    
    def test_invalid_character(self):
        """Test that invalid characters raise an error."""
        source = "int x = @;"
        lexer = Lexer(source)
        
        with self.assertRaises(LexicalError):
            lexer.tokenize()


if __name__ == '__main__':
    unittest.main()
