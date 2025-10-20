# LexicalEchoes

A comprehensive lexical and semantic analyzer for source code analysis.

## Overview

LexicalEchoes is a Python-based tool that performs both lexical and semantic analysis on source code. It tokenizes input code (lexical analysis) and then performs type checking, scope resolution, and semantic validation.

## Features

### Lexical Analysis (Tokenization)
- **Keywords**: `if`, `else`, `while`, `for`, `return`, `int`, `float`, `string`, `bool`, `true`, `false`, `void`
- **Identifiers**: Variable names, function names
- **Literals**: Integer, float, and string literals
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `%`), comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`), logical (`&&`, `||`, `!`), assignment (`=`)
- **Delimiters**: `()`, `{}`, `[]`, `;`, `,`, `.`
- **Comments**: Single-line (`//`) and multi-line (`/* */`)

### Semantic Analysis
- **Symbol Table Management**: Tracks variable declarations and their scopes
- **Type Checking**: Validates type compatibility in assignments and expressions
- **Scope Resolution**: Manages nested scopes
- **Error Detection**: 
  - Duplicate variable declarations
  - Undefined variable usage
  - Type mismatches
  - Uninitialized variables

## Installation

No external dependencies required! LexicalEchoes uses only Python standard library.

```bash
# Clone the repository
git clone https://github.com/hammadrizwan/LexicalEchoes.git
cd LexicalEchoes

# Make main.py executable (optional)
chmod +x main.py
```

## Usage

### Command Line Interface

**Analyze a file:**
```bash
python main.py examples/example1.txt
```

**Analyze with verbose output:**
```bash
python main.py examples/example1.txt -v
```

**Analyze code string:**
```bash
python main.py -c "int x = 5; float y = 3.14;"
```

**Interactive mode:**
```bash
python main.py -i
```

### Examples

The `examples/` directory contains sample source files:

- `example1.txt` - Simple variable declarations
- `example2.txt` - Variable declarations and assignments
- `example3_errors.txt` - Code with semantic errors
- `example4_complex.txt` - More complex examples

### Python API

You can also use LexicalEchoes in your Python code:

```python
from lexer import Lexer
from semantic_analyzer import SemanticAnalyzer

# Source code to analyze
source_code = """
int x = 10;
float y = 3.14;
x = 20;
"""

# Lexical Analysis
lexer = Lexer(source_code)
tokens = lexer.tokenize()

# Print tokens
for token in tokens:
    print(token)

# Semantic Analysis
analyzer = SemanticAnalyzer(tokens)
symbol_table, errors = analyzer.analyze()

# Check results
if errors:
    print("Errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Analysis completed successfully!")
    print(symbol_table)
```

## Architecture

### Components

1. **token_types.py**: Defines token types and the Token class
2. **lexer.py**: Lexical analyzer that converts source code to tokens
3. **symbol_table.py**: Symbol table for managing variable information
4. **semantic_analyzer.py**: Semantic analyzer for type checking and validation
5. **main.py**: Command-line interface and main driver

### Workflow

```
Source Code → Lexer → Tokens → Semantic Analyzer → Symbol Table + Errors
```

## Testing

Run the unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m unittest tests/test_lexer.py
python -m unittest tests/test_semantic_analyzer.py
```

Or run tests directly:

```bash
cd tests
python test_lexer.py
python test_semantic_analyzer.py
```

## Supported Language Syntax

The analyzer supports a C-like syntax with the following constructs:

### Variable Declarations
```c
int x = 10;
float pi = 3.14;
string name = "John";
bool flag = true;
```

### Assignments
```c
x = 20;
name = "Jane";
```

### Comments
```c
// Single-line comment

/* Multi-line
   comment */
```

## Error Handling

LexicalEchoes provides detailed error messages with line and column information:

### Lexical Errors
- Unexpected characters
- Unterminated string literals
- Invalid number formats

### Semantic Errors
- Undefined variables
- Duplicate declarations
- Type mismatches
- Scope violations

## Example Output

```
╔═══════════════════════════════════════════════════════╗
║           LEXICAL ECHOES                              ║
║     Lexical and Semantic Analyzer                     ║
╚═══════════════════════════════════════════════════════╝

[1] Performing Lexical Analysis...
    ✓ Successfully tokenized 16 tokens

[2] Performing Semantic Analysis...
    ✓ Analysis complete. 4 symbols found.

============================================================
ANALYSIS SUMMARY
============================================================
Total Tokens:     16
Total Symbols:    4
Semantic Errors:  0

✓ Analysis completed successfully
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Hammad Rizwan

## Acknowledgments

Built as a demonstration of lexical and semantic analysis techniques used in compiler design.