---
paths:
  - "data/**"
---

# Data safety rules

- NEVER modify, overwrite, or delete files in data/parquet/
- These are the source-of-truth synthetic dataset. All analysis reads from them.
- If you need to create derived data (attribution credits, snapshots), write to outputs/ or a new directory, never back into data/parquet/
- Parquet files are binary. Don't try to read them with cat or head. Use pandas.
