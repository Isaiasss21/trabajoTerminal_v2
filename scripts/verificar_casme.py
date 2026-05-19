# verificar_casme.py  ← corre esto primero

from dataset.casme_dataset import CASMEDataset, CLASS_NAMES
from torch.utils.data import DataLoader, Subset

CASME_ROOT = r"C:\casme\CASME_2\casme_organizado"

# ─── 1. Cargar y revisar distribución ────────────────────────────────────────
print("=" * 50)
print("  Cargando CASME II...")
print("=" * 50)

dataset = CASMEDataset(root=CASME_ROOT, exclude_otros=False)

# ─── 2. Verificar shapes ─────────────────────────────────────────────────────
print("\nVerificando primeras 3 secuencias:")
for i in range(min(3, len(dataset))):
    seq, label = dataset[i]
    folder = dataset.samples[i][0]
    print(f"  [{i}] {folder.parent.parent.name}/{folder.parent.name}/{folder.name}")
    print(f"       shape={seq.shape}  dtype={seq.dtype}  "
          f"label={label} ({CLASS_NAMES[label]})")
    print(f"       min={seq.min():.3f}  max={seq.max():.3f}")

# ─── 3. Verificar LOSO splits ────────────────────────────────────────────────
splits = dataset.get_subject_splits()
print(f"\nSujetos para LOSO: {len(splits)}")
for subj, indices in sorted(splits.items()):
    print(f"  {subj}: {len(indices)} secuencias")

# ─── 4. Verificar DataLoader ─────────────────────────────────────────────────
print("\nProbando DataLoader (1 batch)...")
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
batch_seq, batch_labels = next(iter(loader))

print(f"  Batch shape:  {batch_seq.shape}")   # (4, 15, 3, 64, 64)
print(f"  Labels:       {batch_labels.tolist()}")
print(f"  Emociones:    {[CLASS_NAMES[l] for l in batch_labels.tolist()]}")
print("\n✓ Todo correcto, listo para entrenar")