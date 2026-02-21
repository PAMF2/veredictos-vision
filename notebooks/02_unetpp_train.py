#!/usr/bin/env python3
"""
Alias de execução para manter sequência numérica sem pulo.

Uso recomendado:
  python notebooks/02_unetpp_train.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("03_unetpp_train.py")
    runpy.run_path(str(target), run_name="__main__")
