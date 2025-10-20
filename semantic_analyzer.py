"""
Semantic Analyzer for analyzing the meaning of tokenized code.
Performs type checking, scope resolution, and semantic validation.
"""

from token_types import Token, TokenType
from symbol_table import Symbol, SymbolTable


class SemanticError(Exception):
    """Exception raised for semantic analysis errors."""
    
    def __init__(self, message, line=None, column=None):
        if line is not None and column is not None:
            super().__init__(f"Semantic Error at {line}:{column} - {message}")
        else:
            super().__init__(f"Semantic Error - {message}")
        self.line = line
        self.column = column


class SemanticAnalyzer:
    """Semantic analyzer that performs type checking and scope analysis."""
    
    def __init__(self, tokens):
        """
        Initialize the semantic analyzer.
        
        Args:
            tokens (list[Token]): List of tokens from the lexer
        """
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None
        self.symbol_table = SymbolTable()
        self.errors = []
    
    def advance(self):
        """Move to the next token."""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def peek(self, offset=1):
        """
        Look ahead at the next token without consuming it.
        
        Args:
            offset (int): How many tokens to look ahead
            
        Returns:
            Token or None: The token at the offset position
        """
        peek_pos = self.position + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def report_error(self, message, token=None):
        """
        Report a semantic error.
        
        Args:
            message (str): Error message
            token (Token): Token where the error occurred
        """
        if token:
            error = SemanticError(message, token.line, token.column)
        else:
            error = SemanticError(message)
        self.errors.append(error)
    
    def is_type_keyword(self, token_type):
        """
        Check if a token type is a type keyword.
        
        Args:
            token_type (TokenType): Token type to check
            
        Returns:
            bool: True if it's a type keyword
        """
        return token_type in [TokenType.INT, TokenType.FLOAT, TokenType.STRING, 
                              TokenType.BOOL, TokenType.VOID]
    
    def get_type_name(self, token_type):
        """
        Get the type name from a token type.
        
        Args:
            token_type (TokenType): Token type
            
        Returns:
            str: Type name
        """
        type_map = {
            TokenType.INT: 'int',
            TokenType.FLOAT: 'float',
            TokenType.STRING: 'string',
            TokenType.BOOL: 'bool',
            TokenType.VOID: 'void',
        }
        return type_map.get(token_type, 'unknown')
    
    def check_type_compatibility(self, type1, type2, operation=None):
        """
        Check if two types are compatible for an operation.
        
        Args:
            type1 (str): First type
            type2 (str): Second type
            operation (str): Optional operation being performed
            
        Returns:
            bool: True if types are compatible
        """
        # Same types are always compatible
        if type1 == type2:
            return True
        
        # Numeric types are compatible with each other
        numeric_types = ['int', 'float']
        if type1 in numeric_types and type2 in numeric_types:
            return True
        
        return False
    
    def infer_literal_type(self, token):
        """
        Infer the type of a literal token.
        
        Args:
            token (Token): Literal token
            
        Returns:
            str: Inferred type
        """
        type_map = {
            TokenType.INTEGER_LITERAL: 'int',
            TokenType.FLOAT_LITERAL: 'float',
            TokenType.STRING_LITERAL: 'string',
            TokenType.TRUE: 'bool',
            TokenType.FALSE: 'bool',
        }
        return type_map.get(token.type, 'unknown')
    
    def analyze_declaration(self):
        """
        Analyze a variable declaration statement.
        Example: int x = 5;
        """
        if not self.is_type_keyword(self.current_token.type):
            return
        
        var_type = self.get_type_name(self.current_token.type)
        type_token = self.current_token
        self.advance()
        
        # Expect identifier
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            self.report_error("Expected identifier after type", self.current_token)
            return
        
        var_name = self.current_token.value
        var_token = self.current_token
        self.advance()
        
        # Check if variable already declared in current scope
        existing = self.symbol_table.lookup(var_name, current_scope_only=True)
        if existing:
            self.report_error(
                f"Variable '{var_name}' already declared in this scope at {existing.line}:{existing.column}",
                var_token
            )
        else:
            # Add to symbol table
            symbol = Symbol(var_name, var_type, self.symbol_table.current_scope_level,
                          var_token.line, var_token.column)
            self.symbol_table.insert(symbol)
        
        # Check for initialization
        if self.current_token and self.current_token.type == TokenType.ASSIGN:
            self.advance()
            # Analyze the expression
            expr_type = self.analyze_expression()
            
            # Type check
            if not self.check_type_compatibility(var_type, expr_type):
                self.report_error(
                    f"Type mismatch: cannot assign '{expr_type}' to '{var_type}'",
                    var_token
                )
            
            # Mark as initialized
            symbol = self.symbol_table.lookup(var_name)
            if symbol:
                symbol.initialized = True
        
        # Expect semicolon
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
    
    def analyze_expression(self):
        """
        Analyze an expression and return its type.
        
        Returns:
            str: Type of the expression
        """
        if not self.current_token:
            return 'unknown'
        
        # Handle literals
        if self.current_token.type in [TokenType.INTEGER_LITERAL, TokenType.FLOAT_LITERAL,
                                       TokenType.STRING_LITERAL, TokenType.TRUE, TokenType.FALSE]:
            expr_type = self.infer_literal_type(self.current_token)
            self.advance()
            return expr_type
        
        # Handle identifiers
        if self.current_token.type == TokenType.IDENTIFIER:
            var_name = self.current_token.value
            var_token = self.current_token
            self.advance()
            
            # Check if variable is declared
            symbol = self.symbol_table.lookup(var_name)
            if not symbol:
                self.report_error(f"Undefined variable '{var_name}'", var_token)
                return 'unknown'
            
            return symbol.type
        
        # Handle parenthesized expressions
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            expr_type = self.analyze_expression()
            if self.current_token and self.current_token.type == TokenType.RPAREN:
                self.advance()
            return expr_type
        
        return 'unknown'
    
    def analyze_assignment(self):
        """
        Analyze an assignment statement.
        Example: x = 10;
        """
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            return
        
        var_name = self.current_token.value
        var_token = self.current_token
        self.advance()
        
        if not self.current_token or self.current_token.type != TokenType.ASSIGN:
            return
        
        # Check if variable is declared
        symbol = self.symbol_table.lookup(var_name)
        if not symbol:
            self.report_error(f"Undefined variable '{var_name}'", var_token)
            return
        
        self.advance()
        
        # Analyze expression
        expr_type = self.analyze_expression()
        
        # Type check
        if not self.check_type_compatibility(symbol.type, expr_type):
            self.report_error(
                f"Type mismatch: cannot assign '{expr_type}' to '{symbol.type}'",
                var_token
            )
        
        # Mark as initialized
        symbol.initialized = True
        
        # Expect semicolon
        if self.current_token and self.current_token.type == TokenType.SEMICOLON:
            self.advance()
    
    def analyze(self):
        """
        Perform semantic analysis on all tokens.
        
        Returns:
            tuple: (symbol_table, errors) - Symbol table and list of errors
        """
        while self.current_token and self.current_token.type != TokenType.EOF:
            # Skip semicolons
            if self.current_token.type == TokenType.SEMICOLON:
                self.advance()
                continue
            
            # Try to parse declaration
            if self.is_type_keyword(self.current_token.type):
                self.analyze_declaration()
            # Try to parse assignment
            elif self.current_token.type == TokenType.IDENTIFIER:
                self.analyze_assignment()
            else:
                # Skip unknown tokens
                self.advance()
        
        return self.symbol_table, self.errors
    
    def print_results(self):
        """Print the analysis results."""
        print("\n" + "="*60)
        print("SEMANTIC ANALYSIS RESULTS")
        print("="*60)
        
        print("\n" + str(self.symbol_table))
        
        if self.errors:
            print("\nErrors Found:")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\nNo semantic errors found.")
        
        print("="*60 + "\n")
