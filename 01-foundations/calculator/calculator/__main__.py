"""Entry point: ``python -m calculator``.

With no arguments this starts the REPL. With ``--gui`` it opens the Tk window.
Anything else is treated as an expression to evaluate and print.
"""

import sys

from calculator.repl import run

if __name__ == "__main__":
    if "--gui" in sys.argv[1:]:
        from calculator.gui import main as gui_main

        sys.exit(gui_main())
    sys.exit(run())
