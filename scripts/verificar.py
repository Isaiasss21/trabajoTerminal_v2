import numpy as np
from pathlib import Path
from collections import Counter

datasets = {
    "CASME II": r"C:\Users\iRoti\GitHub\trabajoTerminal_v2\extraction_casme_v2",
    "MEME":     r"C:\Users\iRoti\GitHub\trabajoTerminal_v2\extraction_meme_v2",
    "SMIC":     r"C:\Users\iRoti\GitHub\trabajoTerminal_v2\extraction_smic_v2",
}

for name, path in datasets.items():
    p = Path(path)
    npys = list(p.rglob("*.npy"))
    shapes = Counter()
    emotions = Counter()
    for f in npys:
        arr = np.load(str(f))
        shapes[str(arr.shape)] += 1
        emotions[f.parent.name] += 1
    print(f"\n{'='*40}")
    print(f"{name}: {len(npys)} secuencias")
    print(f"  Shapes: {dict(shapes)}")
    print(f"  Emociones: {dict(sorted(emotions.items()))}")