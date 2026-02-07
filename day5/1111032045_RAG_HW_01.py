import sys
import os
from asyncio import run

_PASS = False
_MSG = ""
try:
    from main_hw import amain

    _PASS = True
except ImportError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
    try:
        from main_hw import amain

        _PASS = True
    except ImportError as e:
        _MSG = f"Import failed. Checked local and {parent_dir}. Error: {e}"

if _PASS:
    if __name__ == "__main__":
        run(amain())
else:
    print(f"Error details: {_MSG}")
