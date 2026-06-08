import importlib.util
import json
import shutil
import sys

python_modules = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "anndata",
    "scanpy",
    "scvi",
    "umap",
    "phate",
    "pacmap",
    "trimap",
    "networkx",
    "community",
]

report = {
    "python": sys.version,
    "executables": {
        "python": shutil.which("python"),
        "R": shutil.which("R"),
        "matlab": shutil.which("matlab"),
        "octave": shutil.which("octave"),
        "conda": shutil.which("conda"),
    },
    "python_modules": {},
}

for name in python_modules:
    report["python_modules"][name] = importlib.util.find_spec(name) is not None

print(json.dumps(report, indent=2, sort_keys=True))
