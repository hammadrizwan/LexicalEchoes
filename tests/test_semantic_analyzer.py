"""
Unit tests for the semantic analyzer module.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lexer import Lexer
from semantic_analyzer import SemanticAnalyzer


class TestSemanticAnalyzer(unittest.TestCase):
    """Test cases for the SemanticAnalyzer class."""
    
    def test_simple_declaration(self):
        """Test simple variable declaration."""
        source = "int x = 10;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(symbol_table.get_all_symbols()), 1)
        
        symbol = symbol_table.lookup('x')
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.type, 'int')
    
    def test_multiple_declarations(self):
        """Test multiple variable declarations."""
        source = """
        int x = 10;
        float y = 3.14;
        string name = "test";
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(symbol_table.get_all_symbols()), 3)
    
    def test_duplicate_declaration(self):
        """Test that duplicate declarations are caught."""
        source = """
        int x = 10;
        int x = 20;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 1)
        self.assertIn("already declared", str(errors[0]))
    
    def test_undefined_variable(self):
        """Test that undefined variables are caught."""
        source = "y = 5;"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 1)
        self.assertIn("Undefined variable", str(errors[0]))
    
    def test_type_mismatch(self):
        """Test that type mismatches are caught."""
        source = 'int x = "hello";'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 1)
        self.assertIn("Type mismatch", str(errors[0]))
    
    def test_assignment_to_declared_variable(self):
        """Test assignment to a declared variable."""
        source = """
        int x = 10;
        x = 20;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 0)
    
    def test_numeric_type_compatibility(self):
        """Test that int and float are compatible."""
        source = """
        int x = 10;
        float y = 3.14;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 0)
    
    def test_boolean_literals(self):
        """Test boolean literal recognition."""
        source = """
        bool flag1 = true;
        bool flag2 = false;
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(symbol_table.get_all_symbols()), 2)


if __name__ == '__main__':
    unittest.main()
