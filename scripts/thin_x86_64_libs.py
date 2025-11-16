from pathlib import Path
import subprocess

SRC = Path('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload')
DST = Path('.venv-x86/lib/python3.10/lib-dynload')
DST.mkdir(parents=True, exist_ok=True)

for so in SRC.glob('*.so'):
    subprocess.run(['lipo', '-thin', 'x86_64', str(so), '-output', str(DST / so.name)], check=True)
