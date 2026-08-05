# Expression Calculator

A calculator that actually parses. Instead of pattern-matching a few shapes of
input, it runs the same three stages a real language implementation does:

```
"2 + 3 * 4"  ->  tokenize  ->  parse  ->  evaluate  ->  14
                 [tokens]      [tree]      [float]
```

Splitting it that way is the whole point of the project. Precedence,
associativity, and error positions all fall out of the grammar rather than
being special-cased.

## Run it

```bash
python -m calculator                # interactive prompt
python -m calculator "2 ^ 10"       # one-shot
python -m calculator --gui          # Tk window
```

The first two need nothing but the standard library. `--gui` additionally needs
Tk, which some distributions package separately from Python — on Arch that is
`sudo pacman -S tk`, on Debian/Ubuntu `sudo apt install python3-tk`.

```
> x = 12
  12
> sqrt(x) * 2
  6.928203230275
> ans ^ 2
  48
```

## What it supports

| | |
|---|---|
| Operators | `+` `-` `*` `/` `//` `%` `^` (`**`) with unary `+`/`-` |
| Grouping | `( )`, arbitrarily nested |
| Variables | `x = 5`, chained `a = b = 3`, self-reference `n = n + 1` |
| Auto-binding | `ans` always holds the previous result |
| Constants | `pi` `e` `tau` `inf` — shadowable |
| Functions | 30 of them: `sqrt` `cbrt` `log` `sin` `hypot` `fact` `gcd` `avg` … |

## The two precedence decisions worth knowing

Most calculator bugs live here, so both are pinned by tests:

```
2 ^ 3 ^ 2   =  512     right-associative, so 2^(3^2) — not (2^3)^2 = 64
-2 ^ 2      =   -4     exponent binds tighter than unary minus: -(2^2)
2 ^ -1      =  0.5     but the exponent may still be negative
```

The grammar gets this right by having `power` recurse into `factor` on its
right-hand side rather than into itself.

## Errors point at the problem

```
> 1 + * 2
  error: unexpected '*'
  1 + * 2
      ^

> sqr(4)
  error: unknown function 'sqr' (did you mean 'sqrt'?)
```

Every token carries its source offset, so the caret is exact rather than
guessed. Nothing escapes as a Python traceback — `EvalError`, `ParseError`, and
`TokenizeError` are the only three things the front-ends catch.

## Layout

```
calculator/
├── tokens.py       text     -> tokens      (what each character run is)
├── ast_nodes.py    node types, no behaviour
├── parser.py       tokens   -> tree        (whether that sequence is legal)
├── evaluator.py    tree     -> float       (what it means)
├── repl.py         terminal front-end
└── gui.py          tkinter front-end
```

The nodes hold no logic, so a second consumer — a pretty-printer, a simplifier
— can walk the same tree without every node type growing another method.

## Tests

```bash
pip install -r requirements.txt
pytest -q          # 100 tests
```

Coverage is concentrated where the bugs are: number lexing edge cases (`.5`,
`1e-3`, `2.`), operator greediness (`//` must not read as two `/`),
associativity, and every reachable error path.
