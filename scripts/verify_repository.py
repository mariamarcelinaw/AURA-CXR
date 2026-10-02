from pathlib import Path
import subprocess,sys
r=Path(__file__).parents[1]; raise SystemExit(subprocess.call([sys.executable,str(r/'aura_cxr.py'),'verify']))
