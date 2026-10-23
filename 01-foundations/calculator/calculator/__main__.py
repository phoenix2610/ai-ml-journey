import sys

from calculator.repl import run

if __name__ == "__main__":
    if "--gui" in sys.argv[1:]:
        from calculator.gui import main as gui_main

        sys.exit(gui_main())
    sys.exit(run())
