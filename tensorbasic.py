#!/usr/bin/env python3
"""
tensorBASIC - A variant of Applesoft BASIC with tensor operations

Features:
- Lowercase keywords (Applesoft BASIC style)
- Tensor arrays with PyTorch backend
- Durable execution with checkpoints
- Broadcasting and slicing support
"""

import sys
import os
import json
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Try to import PyTorch, fall back to NumPy
try:
    import torch
    BACKEND = 'torch'
    print("Using PyTorch backend", file=sys.stderr)
except ImportError:
    BACKEND = 'numpy'
    print("PyTorch not found, using NumPy backend", file=sys.stderr)


class TensorBASICError(Exception):
    """Base exception for tensorBASIC errors"""
    pass


def create_tensor(data, dtype=None):
    """Create a tensor using the active backend"""
    if BACKEND == 'torch':
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(data, dtype=dtype)
    else:
        if dtype is None:
            dtype = np.float32
        return np.array(data, dtype=dtype)


def zeros_tensor(shape):
    """Create a zero tensor using the active backend"""
    if BACKEND == 'torch':
        return torch.zeros(shape)
    else:
        return np.zeros(shape, dtype=np.float32)


def is_tensor(obj):
    """Check if object is a tensor"""
    if BACKEND == 'torch':
        return isinstance(obj, torch.Tensor)
    else:
        return isinstance(obj, np.ndarray)


def save_tensors(tensor_dict, filepath):
    """Save tensors to file"""
    if BACKEND == 'torch':
        torch.save(tensor_dict, filepath)
    else:
        np.savez(filepath, **tensor_dict)


def load_tensors(filepath):
    """Load tensors from file"""
    if BACKEND == 'torch':
        return torch.load(filepath)
    else:
        data = np.load(filepath)
        return {key: data[key] for key in data.files}


class Lexer:
    """Tokenize BASIC source code"""

    TOKEN_PATTERNS = [
        ('NUMBER', r'\d+\.?\d*'),
        ('STRING', r'"[^"]*"'),
        ('KEYWORD', r'\b(let|print|dim|for|next|to|step|if|then|goto|gosub|return|end|rem|input)\b'),
        ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('COMMA', r','),
        ('COLON', r':'),
        ('MATMUL', r'@'),
        ('COMPARE', r'(==|!=|<=|>=|<>|<|>)'),
        ('ASSIGN', r'='),
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('TIMES', r'\*'),
        ('DIVIDE', r'/'),
        ('POWER', r'\^'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'[ \t]+'),
    ]

    def __init__(self, text: str):
        self.text = text
        self.tokens = []
        self._tokenize()

    def _tokenize(self):
        """Convert source text into tokens"""
        pos = 0
        line_num = 1

        while pos < len(self.text):
            match = None

            for token_type, pattern in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.text, pos)

                if match:
                    value = match.group(0)

                    if token_type == 'SKIP':
                        pass  # Ignore whitespace
                    elif token_type == 'NEWLINE':
                        line_num += 1
                    else:
                        self.tokens.append((token_type, value, line_num))

                    pos = match.end()
                    break

            if not match:
                raise TensorBASICError(f"Illegal character at position {pos}: {self.text[pos]}")


class Parser:
    """Parse BASIC source into executable statements"""

    def __init__(self, source: str):
        self.source = source
        self.lines: Dict[int, Dict[str, Any]] = {}
        self._parse()

    def _parse(self):
        """Parse BASIC program into line-numbered statements"""
        for line in self.source.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Extract line number
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if not match:
                continue

            line_num = int(match.group(1))
            statement = match.group(2).strip()

            # Parse statement type
            self.lines[line_num] = self._parse_statement(statement, line_num)

    def _parse_statement(self, statement: str, line_num: int) -> Dict[str, Any]:
        """Parse a single statement"""
        # Split by keyword
        parts = statement.split(None, 1)
        if not parts:
            return {'type': 'empty'}

        keyword = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ''

        return {
            'type': keyword,
            'raw': statement,
            'rest': rest,
            'line': line_num
        }


class ExecutionState:
    """Manages execution state with checkpointing"""

    def __init__(self):
        self.pc: Optional[int] = None  # Program counter
        self.variables: Dict[str, Any] = {}
        self.stack: List[Dict[str, Any]] = []  # For GOSUB and FOR loops
        self.running: bool = False

    def save_checkpoint(self, filepath: str):
        """Save execution state to checkpoint file"""
        checkpoint_data = {
            'pc': self.pc,
            'variables': {},
            'stack': self.stack,
            'running': self.running
        }

        # Separate tensors from regular variables
        tensor_data = {}
        for key, value in self.variables.items():
            if is_tensor(value):
                tensor_data[key] = value
            else:
                checkpoint_data['variables'][key] = value

        # Save JSON metadata
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save tensors separately
        if tensor_data:
            if BACKEND == 'torch':
                tensor_file = filepath + '.tensors'
            else:
                tensor_file = filepath + '.tensors.npz'
            save_tensors(tensor_data, tensor_file)

    def load_checkpoint(self, filepath: str):
        """Load execution state from checkpoint file"""
        with open(filepath, 'r') as f:
            checkpoint_data = json.load(f)

        self.pc = checkpoint_data['pc']
        self.variables = checkpoint_data['variables']
        self.stack = checkpoint_data['stack']
        self.running = checkpoint_data['running']

        # Load tensors if they exist
        if BACKEND == 'torch':
            tensor_file = filepath + '.tensors'
        else:
            tensor_file = filepath + '.tensors.npz'

        if os.path.exists(tensor_file):
            tensor_data = load_tensors(tensor_file)
            self.variables.update(tensor_data)


class TensorBASICInterpreter:
    """Main interpreter for tensorBASIC"""

    def __init__(self, program_file: str, checkpoint_dir: str = './checkpoints'):
        self.program_file = program_file
        self.checkpoint_dir = checkpoint_dir
        self.parser: Optional[Parser] = None
        self.state = ExecutionState()
        self.sorted_lines: List[int] = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def load_program(self):
        """Load BASIC program from file"""
        with open(self.program_file, 'r') as f:
            source = f.read()

        self.parser = Parser(source)
        self.sorted_lines = sorted(self.parser.lines.keys())

    def run(self, resume: bool = False, checkpoint_interval: int = 10):
        """Execute the BASIC program with checkpointing"""
        if resume:
            self._resume_from_checkpoint()
        else:
            self.load_program()
            self.state.pc = self.sorted_lines[0] if self.sorted_lines else None
            self.state.running = True

        step_count = 0

        while self.state.running and self.state.pc is not None:
            # Execute current line
            self._execute_line(self.state.pc)

            # Checkpoint periodically
            step_count += 1
            if step_count % checkpoint_interval == 0:
                self._save_checkpoint()

            # Advance to next line (if not already changed by GOTO/GOSUB)
            if self.state.running and self.state.pc in self.sorted_lines:
                current_idx = self.sorted_lines.index(self.state.pc)
                if current_idx + 1 < len(self.sorted_lines):
                    self.state.pc = self.sorted_lines[current_idx + 1]
                else:
                    self.state.running = False

        # Final checkpoint
        self._save_checkpoint()

    def _execute_line(self, line_num: int):
        """Execute a single line of BASIC"""
        if line_num not in self.parser.lines:
            raise TensorBASICError(f"Line {line_num} not found")

        stmt = self.parser.lines[line_num]
        stmt_type = stmt['type']

        if stmt_type == 'rem':
            pass  # Comment, do nothing

        elif stmt_type == 'print':
            self._execute_print(stmt['rest'])

        elif stmt_type == 'let':
            self._execute_let(stmt['rest'])

        elif stmt_type == 'dim':
            self._execute_dim(stmt['rest'])

        elif stmt_type == 'goto':
            target = int(stmt['rest'].strip())
            self.state.pc = target

        elif stmt_type == 'end':
            self.state.running = False

        elif stmt_type == 'for':
            self._execute_for(stmt['rest'], line_num)

        elif stmt_type == 'next':
            self._execute_next(stmt['rest'])

        elif stmt_type == 'if':
            self._execute_if(stmt['rest'])

        elif stmt_type == 'gosub':
            target = int(stmt['rest'].strip())
            self.stack.append({'type': 'gosub', 'return_pc': self.state.pc})
            self.state.pc = target

        elif stmt_type == 'return':
            if not self.state.stack or self.state.stack[-1]['type'] != 'gosub':
                raise TensorBASICError("RETURN without GOSUB")
            frame = self.state.stack.pop()
            self.state.pc = frame['return_pc']

    def _execute_print(self, expr: str):
        """Execute PRINT statement"""
        value = self._eval_expr(expr)
        print(value)

    def _execute_let(self, expr: str):
        """Execute LET statement (assignment)"""
        # Parse: variable = expression
        match = re.match(r'(\w+(?:\[.+\])?)\s*=\s*(.+)', expr)
        if not match:
            raise TensorBASICError(f"Invalid LET statement: {expr}")

        var_part = match.group(1)
        value_expr = match.group(2)

        # Check if this is array indexing
        if '[' in var_part:
            var_match = re.match(r'(\w+)\[(.+)\]', var_part)
            var_name = var_match.group(1)
            indices = var_match.group(2)

            # Evaluate indices and value
            idx = self._parse_indices(indices)
            value = self._eval_expr(value_expr)

            # Assign to tensor slice
            if var_name not in self.state.variables:
                raise TensorBASICError(f"Undefined variable: {var_name}")

            self.state.variables[var_name][idx] = value
        else:
            # Simple assignment
            value = self._eval_expr(value_expr)
            self.state.variables[var_part] = value

    def _execute_dim(self, expr: str):
        """Execute DIM statement (array/tensor declaration)"""
        # Parse: array(dim1, dim2, ...)
        match = re.match(r'(\w+)\(([^)]+)\)', expr)
        if not match:
            raise TensorBASICError(f"Invalid DIM statement: {expr}")

        var_name = match.group(1)
        dims_str = match.group(2)

        # Parse dimensions
        dims = [int(d.strip()) for d in dims_str.split(',')]

        # Create tensor
        self.state.variables[var_name] = zeros_tensor(dims)

    def _execute_for(self, expr: str, line_num: int):
        """Execute FOR statement"""
        # Parse: i = start to end [step increment]
        match = re.match(r'(\w+)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+))?$', expr)
        if not match:
            raise TensorBASICError(f"Invalid FOR statement: {expr}")

        var_name = match.group(1)
        start = self._eval_expr(match.group(2))
        end = self._eval_expr(match.group(3))
        step = self._eval_expr(match.group(4)) if match.group(4) else 1

        # Initialize loop variable
        self.state.variables[var_name] = start

        # Push loop frame onto stack
        self.state.stack.append({
            'type': 'for',
            'var': var_name,
            'end': end,
            'step': step,
            'start_line': line_num
        })

    def _execute_next(self, var_name: str):
        """Execute NEXT statement"""
        var_name = var_name.strip()

        if not self.state.stack or self.state.stack[-1]['type'] != 'for':
            raise TensorBASICError("NEXT without FOR")

        frame = self.state.stack[-1]

        if frame['var'] != var_name:
            raise TensorBASICError(f"NEXT {var_name} doesn't match FOR {frame['var']}")

        # Increment loop variable
        self.state.variables[var_name] += frame['step']

        # Check if loop should continue
        if frame['step'] > 0:
            done = self.state.variables[var_name] > frame['end']
        else:
            done = self.state.variables[var_name] < frame['end']

        if done:
            # Exit loop
            self.state.stack.pop()
        else:
            # Jump back to FOR line
            self.state.pc = frame['start_line']

    def _execute_if(self, expr: str):
        """Execute IF statement"""
        # Parse: condition then action
        match = re.match(r'(.+?)\s+then\s+(.+)', expr)
        if not match:
            raise TensorBASICError(f"Invalid IF statement: {expr}")

        condition = match.group(1)
        action = match.group(2)

        # Evaluate condition
        result = self._eval_expr(condition)

        if result:
            # Check if action is a line number (GOTO) or statement
            if action.strip().isdigit():
                self.state.pc = int(action.strip())
            else:
                # Execute statement inline
                stmt = self._parse_statement(action, self.state.pc)
                self._execute_line_from_stmt(stmt)

    def _execute_line_from_stmt(self, stmt: Dict[str, Any]):
        """Execute a statement dict (for inline IF statements)"""
        # This is a simplified version - could be expanded
        if stmt['type'] == 'print':
            self._execute_print(stmt['rest'])

    def _eval_expr(self, expr: str) -> Any:
        """Evaluate an expression"""
        expr = expr.strip()

        # String literal
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]

        # List/tensor literal
        if expr.startswith('['):
            return self._parse_tensor_literal(expr)

        # Variable or numeric
        try:
            # Try numeric
            if '.' in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            # Must be variable or expression
            pass

        # Variable reference with indexing
        if '[' in expr and ']' in expr:
            match = re.match(r'(\w+)\[(.+)\]', expr)
            if match:
                var_name = match.group(1)
                indices = match.group(2)

                if var_name not in self.state.variables:
                    raise TensorBASICError(f"Undefined variable: {var_name}")

                idx = self._parse_indices(indices)
                return self.state.variables[var_name][idx]

        # Simple variable
        if expr in self.state.variables:
            return self.state.variables[expr]

        # Try to evaluate as Python expression (for operators)
        try:
            # Build namespace with variables
            namespace = dict(self.state.variables)
            # Handle @ as matmul
            expr = expr.replace('@', ' @ ')
            return eval(expr, {"__builtins__": {}}, namespace)
        except:
            raise TensorBASICError(f"Cannot evaluate expression: {expr}")

    def _parse_tensor_literal(self, expr: str):
        """Parse tensor literal like [[1,2],[3,4]]"""
        try:
            data = eval(expr)
            return create_tensor(data)
        except:
            raise TensorBASICError(f"Invalid tensor literal: {expr}")

    def _parse_indices(self, indices: str) -> Union[int, slice, Tuple]:
        """Parse array indices like '0:2, 1:3' or '0, 1'"""
        parts = [p.strip() for p in indices.split(',')]
        result = []

        for part in parts:
            if ':' in part:
                # Slice notation
                slice_parts = part.split(':')
                start = int(slice_parts[0]) if slice_parts[0] else None
                stop = int(slice_parts[1]) if slice_parts[1] else None
                step = int(slice_parts[2]) if len(slice_parts) > 2 and slice_parts[2] else None
                result.append(slice(start, stop, step))
            else:
                # Single index
                result.append(int(part))

        return tuple(result) if len(result) > 1 else result[0]

    def _save_checkpoint(self):
        """Save current execution state"""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            os.path.basename(self.program_file) + '.checkpoint'
        )
        self.state.save_checkpoint(checkpoint_file)

    def _resume_from_checkpoint(self):
        """Resume execution from checkpoint"""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            os.path.basename(self.program_file) + '.checkpoint'
        )

        if not os.path.exists(checkpoint_file):
            raise TensorBASICError(f"No checkpoint found: {checkpoint_file}")

        self.load_program()
        self.state.load_checkpoint(checkpoint_file)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='tensorBASIC interpreter')
    parser.add_argument('program', help='BASIC program file (.bas)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Lines between checkpoints')

    args = parser.parse_args()

    interpreter = TensorBASICInterpreter(args.program, args.checkpoint_dir)

    try:
        interpreter.run(resume=args.resume, checkpoint_interval=args.checkpoint_interval)
        print("\nProgram completed successfully")
    except TensorBASICError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
