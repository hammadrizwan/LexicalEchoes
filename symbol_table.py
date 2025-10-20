"""
Symbol Table for semantic analysis.
Manages variable declarations, scopes, and type information.
"""


class Symbol:
    """Represents a symbol (variable, function, etc.) in the symbol table."""
    
    def __init__(self, name, symbol_type, scope_level, line, column):
        """
        Initialize a symbol.
        
        Args:
            name (str): Name of the symbol
            symbol_type (str): Type of the symbol (int, float, string, etc.)
            scope_level (int): Scope level where the symbol is declared
            line (int): Line number where declared
            column (int): Column number where declared
        """
        self.name = name
        self.type = symbol_type
        self.scope_level = scope_level
        self.line = line
        self.column = column
        self.initialized = False
    
    def __repr__(self):
        """String representation of the symbol."""
        return f"Symbol({self.name}, {self.type}, scope={self.scope_level}, {self.line}:{self.column})"


class SymbolTable:
    """
    Symbol table for managing symbols across different scopes.
    Supports nested scopes and scope management.
    """
    
    def __init__(self):
        """Initialize the symbol table."""
        self.scopes = [{}]  # List of dictionaries, each representing a scope
        self.current_scope_level = 0
    
    def enter_scope(self):
        """Enter a new scope (e.g., entering a function or block)."""
        self.scopes.append({})
        self.current_scope_level += 1
    
    def exit_scope(self):
        """Exit the current scope."""
        if self.current_scope_level > 0:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def insert(self, symbol):
        """
        Insert a symbol into the current scope.
        
        Args:
            symbol (Symbol): The symbol to insert
            
        Returns:
            bool: True if inserted successfully, False if already exists in current scope
        """
        current_scope = self.scopes[self.current_scope_level]
        if symbol.name in current_scope:
            return False
        current_scope[symbol.name] = symbol
        return True
    
    def lookup(self, name, current_scope_only=False):
        """
        Look up a symbol by name.
        
        Args:
            name (str): Name of the symbol to look up
            current_scope_only (bool): If True, only search in current scope
            
        Returns:
            Symbol or None: The symbol if found, None otherwise
        """
        if current_scope_only:
            return self.scopes[self.current_scope_level].get(name)
        
        # Search from current scope to outer scopes
        for i in range(self.current_scope_level, -1, -1):
            if name in self.scopes[i]:
                return self.scopes[i][name]
        return None
    
    def get_all_symbols(self):
        """
        Get all symbols from all scopes.
        
        Returns:
            list[Symbol]: List of all symbols
        """
        symbols = []
        for scope in self.scopes:
            symbols.extend(scope.values())
        return symbols
    
    def __str__(self):
        """String representation of the symbol table."""
        result = "Symbol Table:\n"
        for level, scope in enumerate(self.scopes):
            result += f"  Scope {level}:\n"
            for name, symbol in scope.items():
                result += f"    {symbol}\n"
        return result
