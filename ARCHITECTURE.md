# LexicalEchoes Architecture

## Overview

LexicalEchoes is a complete lexical and semantic analyzer implementation that demonstrates compiler design principles. The system is divided into several modular components that work together to analyze source code.

## System Components

### 1. Token Types (`token_types.py`)

**Purpose**: Define all possible token types and provide the Token class.

**Key Classes**:
- `TokenType` (Enum): Enumeration of all token types including:
  - Keywords (if, else, while, for, return, int, float, string, bool, true, false, void)
  - Identifiers and literals
  - Operators (arithmetic, comparison, logical)
  - Delimiters (parentheses, braces, brackets, semicolons, etc.)
  
- `Token`: Represents a single token with:
  - Type (TokenType)
  - Value (lexeme)
  - Position (line and column numbers)

**Data Structure**: `KEYWORDS` dictionary for fast keyword lookup.

### 2. Lexical Analyzer (`lexer.py`)

**Purpose**: Convert raw source code into a stream of tokens.

**Key Classes**:
- `LexicalError`: Custom exception for lexical errors
- `Lexer`: Main lexer class

**Algorithm**:
```
1. Initialize with source code
2. Track current position, line, and column
3. For each character:
   a. Skip whitespace and comments
   b. Recognize numbers (integers and floats)
   c. Recognize strings (with escape sequences)
   d. Recognize identifiers and keywords
   e. Recognize operators (both single and double-character)
   f. Recognize delimiters
4. Return stream of tokens
```

**Key Methods**:
- `advance()`: Move to next character
- `peek()`: Look ahead without consuming
- `skip_whitespace()`: Skip space, tab, carriage return
- `skip_comment()`: Handle // and /* */ comments
- `read_number()`: Parse integer and float literals
- `read_string()`: Parse string literals with escape sequences
- `read_identifier()`: Parse identifiers and keywords
- `get_next_token()`: Get next token from source
- `tokenize()`: Tokenize entire source code

**Features**:
- Line and column tracking for error reporting
- Support for escape sequences in strings (\n, \t, \r, \", \\)
- Single-line (//) and multi-line (/* */) comment support
- Two-character operator recognition (==, !=, <=, >=, &&, ||)

### 3. Symbol Table (`symbol_table.py`)

**Purpose**: Manage symbols (variables, functions) and their scopes.

**Key Classes**:
- `Symbol`: Represents a symbol with:
  - Name
  - Type (int, float, string, bool)
  - Scope level
  - Position (line, column)
  - Initialization status

- `SymbolTable`: Manages symbols across scopes
  - Uses list of dictionaries for scope management
  - Supports nested scopes

**Data Structure**: 
```
scopes = [
    {name1: Symbol1, name2: Symbol2},  # Scope 0 (global)
    {name3: Symbol3},                  # Scope 1
    ...
]
```

**Key Methods**:
- `enter_scope()`: Create new nested scope
- `exit_scope()`: Return to parent scope
- `insert()`: Add symbol to current scope
- `lookup()`: Find symbol (searches from current to outer scopes)
- `get_all_symbols()`: Get all symbols from all scopes

### 4. Semantic Analyzer (`semantic_analyzer.py`)

**Purpose**: Perform semantic analysis including type checking and scope resolution.

**Key Classes**:
- `SemanticError`: Custom exception for semantic errors
- `SemanticAnalyzer`: Main analyzer class

**Algorithm**:
```
1. Initialize with token stream
2. Create symbol table
3. For each statement:
   a. Check if it's a declaration
      - Verify type is valid
      - Check for duplicate names in current scope
      - Add to symbol table
      - If initialized, check type compatibility
   b. Check if it's an assignment
      - Verify variable is declared
      - Check type compatibility
      - Mark as initialized
4. Collect all errors (don't stop on first error)
5. Return symbol table and error list
```

**Key Methods**:
- `analyze_declaration()`: Process variable declarations
- `analyze_assignment()`: Process assignments
- `analyze_expression()`: Analyze expressions and infer types
- `check_type_compatibility()`: Verify type compatibility
- `infer_literal_type()`: Determine type of literals
- `is_type_keyword()`: Check if token is a type keyword
- `analyze()`: Main analysis entry point

**Features**:
- Type checking for assignments
- Duplicate declaration detection
- Undefined variable detection
- Type compatibility checking (int/float are compatible)
- Scope-aware symbol resolution
- Multiple error collection (continues analysis after errors)

### 5. Main Driver (`main.py`)

**Purpose**: Provide command-line interface and orchestrate analysis.

**Modes**:
1. **File Mode**: Analyze a source file
2. **Code String Mode**: Analyze code from command line
3. **Interactive Mode**: Enter code interactively

**Key Functions**:
- `print_banner()`: Display application banner
- `print_tokens()`: Format and display tokens
- `analyze_file()`: Analyze a file
- `analyze_string()`: Analyze a code string
- `main()`: Entry point with argument parsing

**Output Format**:
```
1. Banner
2. Lexical Analysis Results
3. Semantic Analysis Results
4. Summary (tokens, symbols, errors)
5. Status (success or errors)
```

## Data Flow

```
Source Code
    ↓
┌─────────────────┐
│     Lexer       │ → LexicalError (if invalid syntax)
└────────┬────────┘
         ↓
    Token Stream
         ↓
┌─────────────────┐
│    Semantic     │ → SemanticError (if semantic issues)
│    Analyzer     │
└────────┬────────┘
         ↓
  Symbol Table + Errors
         ↓
┌─────────────────┐
│     Output      │
│   (Console)     │
└─────────────────┘
```

## Error Handling

### Lexical Errors
- **Unexpected Character**: Unrecognized character in source
- **Unterminated String**: String literal without closing quote
- **Invalid Number**: Malformed numeric literal

### Semantic Errors
- **Duplicate Declaration**: Variable declared twice in same scope
- **Undefined Variable**: Using variable before declaration
- **Type Mismatch**: Incompatible types in assignment
- **Uninitialized Variable**: (tracked but not enforced)

All errors include line and column information for precise debugging.

## Testing Strategy

### Unit Tests

1. **Lexer Tests** (`tests/test_lexer.py`):
   - Keyword recognition
   - Identifier recognition
   - Literal recognition (int, float, string)
   - Operator recognition
   - Delimiter recognition
   - Comment handling
   - Line/column tracking
   - Error handling

2. **Semantic Analyzer Tests** (`tests/test_semantic_analyzer.py`):
   - Simple declarations
   - Multiple declarations
   - Duplicate detection
   - Undefined variable detection
   - Type mismatch detection
   - Assignment validation
   - Type compatibility
   - Boolean literals

### Example Files

1. `example1.txt`: Basic declarations
2. `example2.txt`: Declarations and assignments
3. `example3_errors.txt`: Code with errors
4. `example4_complex.txt`: Complex scenarios

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Error Recovery**: Continue analysis after errors to find all issues
3. **Extensibility**: Easy to add new token types, keywords, or analysis rules
4. **Modularity**: Components can be used independently
5. **Clarity**: Clear error messages with precise location information

## Future Enhancements

Possible extensions to the system:

1. **Parser**: Add syntax analysis (AST generation)
2. **More Types**: Arrays, structs, pointers
3. **Functions**: Function declarations and calls
4. **Control Flow**: If statements, loops, etc.
5. **Advanced Type Checking**: Implicit conversions, type inference
6. **Optimization**: Constant folding, dead code elimination
7. **Code Generation**: Generate intermediate or target code
8. **IDE Integration**: Language server protocol support

## Performance Characteristics

- **Time Complexity**:
  - Lexer: O(n) where n is length of source code
  - Semantic Analyzer: O(m) where m is number of tokens
  - Symbol Lookup: O(d) where d is scope depth
  
- **Space Complexity**:
  - Token storage: O(m) where m is number of tokens
  - Symbol table: O(s) where s is number of symbols

## Conclusion

LexicalEchoes provides a clean, modular implementation of lexical and semantic analysis that can serve as:
- Learning resource for compiler construction
- Foundation for more advanced compiler projects
- Code analysis tool for simple languages
- Demonstration of language processing techniques
