# tensorBASIC

A variant of Applesoft BASIC where arrays are tensors, with built-in tensor operations and a durable execution model.

## Features

- **Lowercase Keywords**: Applesoft BASIC style with lowercase keywords
- **Tensor Arrays**: Multi-dimensional arrays powered by PyTorch
- **Tensor Operations**: Built-in support for slicing, broadcasting, and matrix operations
- **Durable Execution**: Checkpoint-based execution model that saves program state
- **Resumable Programs**: Resume execution from checkpoints after interruption

## Prerequisites

- Python 3.6+
- PyTorch 2.0+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running a Program

```bash
python tensorbasic.py program.bas
```

### Resuming from Checkpoint

```bash
python tensorbasic.py program.bas --resume
```

### Checkpoint Options

```bash
# Specify checkpoint directory
python tensorbasic.py program.bas --checkpoint-dir ./my_checkpoints

# Change checkpoint interval (default: every 10 lines)
python tensorbasic.py program.bas --checkpoint-interval 20
```

## Language Reference

### Basic Structure

Programs consist of numbered lines executed sequentially:

```basic
10 rem This is a comment
20 print "Hello, tensorBASIC!"
30 end
```

### Data Types

- **Numbers**: Integers and floats (`42`, `3.14`)
- **Strings**: Double-quoted text (`"hello"`)
- **Tensors**: Multi-dimensional arrays

### Variables

```basic
10 let x = 42
20 let name = "Alice"
30 let pi = 3.14159
```

### Tensor Arrays

#### Declaration

```basic
10 dim matrix(3,3)        rem 3x3 matrix
20 dim tensor(2,3,4)      rem 2x3x4 tensor
```

#### Initialization

```basic
10 let a = [[1,2,3],[4,5,6],[7,8,9]]
```

#### Indexing and Slicing

```basic
10 dim a(5,5)
20 let a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
30 rem Get element at row 0, col 1
40 let x = a[0, 1]
50 rem Slice rows 0-2, cols 1-3
60 let b = a[0:2, 1:3]
70 print b
```

### Tensor Operations

#### Matrix Multiplication

```basic
10 dim a(3,3)
20 dim b(3,3)
30 let a = [[1,2,3],[4,5,6],[7,8,9]]
40 let b = [[9,8,7],[6,5,4],[3,2,1]]
50 let c = a @ b          rem Matrix multiply
60 print c
```

#### Element-wise Operations

```basic
10 let sum = a + b        rem Element-wise addition
20 let diff = a - b       rem Element-wise subtraction
30 let prod = a * b       rem Element-wise multiplication
40 let quot = a / b       rem Element-wise division
```

### Control Flow

#### Conditionals

```basic
10 let x = 10
20 if x > 5 then print "x is greater than 5"
30 if x == 10 then 100
40 print "x is not 10"
50 goto 110
100 print "x is 10"
110 end
```

Comparison operators: `<`, `>`, `<=`, `>=`, `==`, `!=`, `<>`

#### Loops

```basic
10 for i = 1 to 10 step 1
20   print i
30 next i

40 for i = 10 to 1 step -1
50   print i
60 next i
```

#### Unconditional Jump

```basic
10 let x = 0
20 print x
30 let x = x + 1
40 if x < 10 then 20
50 end
```

### Subroutines

```basic
10 gosub 100
20 print "Back in main"
30 end
100 rem Subroutine
110 print "In subroutine"
120 return
```

### Input/Output

```basic
10 print "Hello, World!"
20 print x
30 print a                 rem Print tensor
```

### Statements Reference

| Statement | Syntax | Description |
|-----------|--------|-------------|
| `rem` | `rem comment` | Comment (ignored) |
| `let` | `let var = expr` | Variable assignment |
| `print` | `print expr` | Output to console |
| `dim` | `dim name(d1,d2,...)` | Declare tensor array |
| `if` | `if cond then action` | Conditional execution |
| `for` | `for var = start to end step inc` | Loop start |
| `next` | `next var` | Loop end |
| `goto` | `goto linenum` | Jump to line |
| `gosub` | `gosub linenum` | Call subroutine |
| `return` | `return` | Return from subroutine |
| `end` | `end` | End program |

## Examples

### Hello World

```basic
10 rem Hello World in tensorBASIC
20 print "Hello, tensorBASIC!"
30 end
```

### Matrix Operations

```basic
10 rem Matrix multiplication example
20 dim a(3,3)
30 dim b(3,3)
40 let a = [[1,2,3],[4,5,6],[7,8,9]]
50 let b = [[9,8,7],[6,5,4],[3,2,1]]
60 let c = a @ b
70 print "Result:"
80 print c
90 end
```

### Neural Network Forward Pass

```basic
10 rem Simple neural network
20 dim x(2,4)
30 let x = [[1,2,3,4],[5,6,7,8]]
40 dim w1(4,3)
50 let w1 = [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[1.0,1.1,1.2]]
60 let hidden = x @ w1
70 print "Hidden layer:"
80 print hidden
90 end
```

More examples available in the `examples/` directory:
- `examples/hello.bas` - Basic hello world
- `examples/matrix_ops.bas` - Matrix operations
- `examples/loops.bas` - Loop examples
- `examples/conditionals.bas` - Conditional statements
- `examples/subroutines.bas` - Subroutine usage
- `examples/neural_net.bas` - Neural network forward pass

## Durable Execution Model

tensorBASIC automatically saves checkpoints during execution:

- **Checkpoint Frequency**: Configurable (default: every 10 lines)
- **Checkpoint Contents**:
  - Program counter (current line)
  - All variables (including tensors)
  - Call stack (GOSUB frames)
  - Loop stack (FOR frames)

### Checkpoint Files

Checkpoints are saved as:
- `program.bas.checkpoint` - JSON metadata
- `program.bas.checkpoint.tensors` - PyTorch tensor data

### Resume Example

```bash
# Start program
python tensorbasic.py long_computation.bas

# If interrupted, resume from last checkpoint
python tensorbasic.py long_computation.bas --resume
```

## Architecture

- **Lexer**: Tokenizes BASIC source code
- **Parser**: Converts source into line-numbered statements
- **Execution Engine**: Interprets statements with program counter
- **State Management**: Handles variables, stack, and checkpoints
- **Tensor Backend**: PyTorch for array operations

## Future Enhancements

Potential features for future development:

- [ ] More tensor operations (reshape, transpose, etc.)
- [ ] Built-in functions (sin, cos, exp, log, etc.)
- [ ] String operations
- [ ] File I/O
- [ ] INPUT statement for user input
- [ ] DATA/READ statements
- [ ] Array bounds checking
- [ ] Error handling and debugging features
- [ ] GPU acceleration support
- [ ] Distributed checkpointing
- [ ] REPL mode

## License

MIT License (inherited from original pytorch-cifar repository)

## Credits

Originally based on the pytorch-cifar repository, now repurposed as a tensorBASIC prototype.
