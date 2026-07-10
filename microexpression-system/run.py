# run.py — punto de entrada para PyInstaller
# No modificar: es el launcher que PyInstaller usa como __main__.
import sys
from app.main import main

sys.exit(main())
