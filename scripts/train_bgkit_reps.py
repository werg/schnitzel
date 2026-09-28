"""Compatibility name for ``scripts/train.py writer`` (launch scripts of B3/B4 use it)."""
from schnitz.kb.stages.writer import main

if __name__ == '__main__':
    main()
