#!/usr/bin/env python3
"""
LexicalEchoes - Main driver program for lexical and semantic analysis.
Provides a command-line interface to analyze source code.
"""

import sys
import argparse
from lexer import Lexer, LexicalError
from semantic_analyzer import SemanticAnalyzer


def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║           LEXICAL ECHOES                              ║
    ║     Lexical and Semantic Analyzer                     ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_tokens(tokens):
    """
    Print tokens in a formatted way.
    
    Args:
        tokens (list[Token]): List of tokens to print
    """
    print("\n" + "="*60)
    print("LEXICAL ANALYSIS - TOKENS")
    print("="*60)
    print(f"{'Line:Col':<12} {'Token Type':<20} {'Value':<20}")
    print("-"*60)
    
    for token in tokens:
        location = f"{token.line}:{token.column}"
        value = repr(token.value) if token.value else ''
        print(f"{location:<12} {token.type.name:<20} {value:<20}")
    
    print("="*60)
    print(f"Total tokens: {len(tokens)}\n")


def analyze_file(filepath, verbose=False):
    """
    Analyze a source code file.
    
    Args:
        filepath (str): Path to the source file
        verbose (bool): Whether to print detailed output
    """
    try:
        # Read source code
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        if verbose:
            print(f"\nSource Code from {filepath}:")
            print("-"*60)
            print(source_code)
            print("-"*60)
        
        # Lexical Analysis
        print("\n[1] Performing Lexical Analysis...")
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        if verbose:
            print_tokens(tokens)
        else:
            print(f"    ✓ Successfully tokenized {len(tokens)} tokens")
        
        # Semantic Analysis
        print("\n[2] Performing Semantic Analysis...")
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        if verbose or errors:
            analyzer.print_results()
        else:
            print(f"    ✓ Analysis complete. {len(symbol_table.get_all_symbols())} symbols found.")
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Tokens:     {len(tokens)}")
        print(f"Total Symbols:    {len(symbol_table.get_all_symbols())}")
        print(f"Semantic Errors:  {len(errors)}")
        
        if errors:
            print("\n⚠ Analysis completed with errors")
            return 1
        else:
            print("\n✓ Analysis completed successfully")
            return 0
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return 1
    except LexicalError as e:
        print(f"Lexical Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def analyze_string(source_code, verbose=False):
    """
    Analyze a source code string.
    
    Args:
        source_code (str): Source code to analyze
        verbose (bool): Whether to print detailed output
    """
    try:
        if verbose:
            print("\nSource Code:")
            print("-"*60)
            print(source_code)
            print("-"*60)
        
        # Lexical Analysis
        print("\n[1] Performing Lexical Analysis...")
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        if verbose:
            print_tokens(tokens)
        else:
            print(f"    ✓ Successfully tokenized {len(tokens)} tokens")
        
        # Semantic Analysis
        print("\n[2] Performing Semantic Analysis...")
        analyzer = SemanticAnalyzer(tokens)
        symbol_table, errors = analyzer.analyze()
        
        if verbose or errors:
            analyzer.print_results()
        else:
            print(f"    ✓ Analysis complete. {len(symbol_table.get_all_symbols())} symbols found.")
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Tokens:     {len(tokens)}")
        print(f"Total Symbols:    {len(symbol_table.get_all_symbols())}")
        print(f"Semantic Errors:  {len(errors)}")
        
        if errors:
            print("\n⚠ Analysis completed with errors")
            return 1
        else:
            print("\n✓ Analysis completed successfully")
            return 0
        
    except LexicalError as e:
        print(f"Lexical Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='LexicalEchoes - Lexical and Semantic Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.txt                  # Analyze a file
  %(prog)s input.txt -v               # Analyze with verbose output
  %(prog)s -c "int x = 5;"            # Analyze code string
  %(prog)s -i                         # Interactive mode
        """
    )
    
    parser.add_argument('file', nargs='?', help='Source file to analyze')
    parser.add_argument('-c', '--code', help='Source code string to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Interactive mode
    if args.interactive:
        print("Interactive Mode - Enter your code (type 'END' on a new line to finish):\n")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        
        source_code = '\n'.join(lines)
        return analyze_string(source_code, args.verbose)
    
    # Analyze code string
    elif args.code:
        return analyze_string(args.code, args.verbose)
    
    # Analyze file
    elif args.file:
        return analyze_file(args.file, args.verbose)
    
    # No input provided
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
