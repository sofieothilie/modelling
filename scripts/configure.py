import sys
compiler = "cl.exe" if sys.platform == "win32" else "gcc"
print(compiler + " /DLL /nologo /MD")