"""Tkinter front-end.

Deliberately thin: every keypress edits a string, and that string goes through
exactly the same parser and evaluator the REPL uses. The GUI owns no maths.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from calculator.evaluator import Environment, EvalError, evaluate, format_result
from calculator.parser import ParseError
from calculator.tokens import TokenizeError

# (label, what it inserts) laid out row by row; None means a special action.
KEYS: list[list[tuple[str, str | None]]] = [
    [("C", None), ("(", "("), (")", ")"), ("/", "/")],
    [("7", "7"), ("8", "8"), ("9", "9"), ("*", "*")],
    [("4", "4"), ("5", "5"), ("6", "6"), ("-", "-")],
    [("1", "1"), ("2", "2"), ("3", "3"), ("+", "+")],
    [("0", "0"), (".", "."), ("^", "^"), ("=", None)],
]

FUNCTION_KEYS = [("sqrt", "sqrt("), ("pi", "pi"), ("ans", "ans"), ("<-", None)]


class CalculatorWindow(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.env = Environment()
        self.entry_var = tk.StringVar()
        self.result_var = tk.StringVar(value="0")
        self._build()
        self.grid(sticky="nsew")

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        self.master.title("Calculator")
        self.master.minsize(300, 400)
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        result = ttk.Label(
            self, textvariable=self.result_var, anchor="e", font=("TkDefaultFont", 24, "bold")
        )
        result.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))

        entry = ttk.Entry(self, textvariable=self.entry_var, justify="right", font=("TkFixedFont", 14))
        entry.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        entry.focus_set()
        entry.bind("<Return>", lambda _e: self.compute())
        entry.bind("<KP_Enter>", lambda _e: self.compute())
        entry.bind("<Escape>", lambda _e: self.clear())

        for col, (label, insert) in enumerate(FUNCTION_KEYS):
            action = self.backspace if insert is None else (lambda s=insert: self.insert(s))
            ttk.Button(self, text=label, width=6, command=action).grid(
                row=2, column=col, sticky="nsew", padx=2, pady=2
            )

        for r, row in enumerate(KEYS, start=3):
            for c, (label, insert) in enumerate(row):
                if insert is None:
                    action = self.clear if label == "C" else self.compute
                else:
                    action = lambda s=insert: self.insert(s)
                ttk.Button(self, text=label, command=action).grid(
                    row=r, column=c, sticky="nsew", padx=2, pady=2
                )

        for c in range(4):
            self.columnconfigure(c, weight=1)
        for r in range(2, 3 + len(KEYS)):
            self.rowconfigure(r, weight=1)

    # ----------------------------------------------------------------- actions

    def insert(self, text: str) -> None:
        self.entry_var.set(self.entry_var.get() + text)

    def backspace(self) -> None:
        self.entry_var.set(self.entry_var.get()[:-1])

    def clear(self) -> None:
        self.entry_var.set("")
        self.result_var.set("0")

    def compute(self) -> None:
        source = self.entry_var.get().strip()
        if not source:
            return
        try:
            value = evaluate(source, self.env)
        except (EvalError, ParseError, TokenizeError) as exc:
            # Errors are multi-line (they carry a caret); the display is one line.
            self.result_var.set(str(exc).splitlines()[0])
            return
        self.env.variables["ans"] = value
        self.result_var.set(format_result(value))


def main() -> int:
    root = tk.Tk()
    CalculatorWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
