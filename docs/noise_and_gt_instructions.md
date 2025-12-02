
# Instructions: Adding Gaussian Noise to Replicated Vectors and Regenerating GT

This document explains **exactly** how to:
1. Add Gaussian noise to replicated vectors in a scaled dataset.
2. Regenerate the ground-truth (GT) so it is consistent with the new, jittered dataset.

It assumes that a developer already implemented **vector replication without noise** and needs to extend it correctly.

---

## 1. Terminology and Parameters

- **N** – number of vectors in the original dataset.
- **D** – dimensionality (number of features) of each vector.
- **S** – scale factor (replication factor). Example: `S = 10` means we produce 10 copies (shards) of the dataset.
- **Q** – number of queries in the GT file.
- **K** – top-K neighbors per query in the original GT.
- **noise** – standard deviation (σ) of the Gaussian noise (e.g. `1e-3`).

Data layout assumptions:

- Dataset is stored in `.fvecs` or `.fbin`:
  - `.fvecs`: each vector stored as `int32 dim + float32[dim]`.
  - `.fbin`: raw float32 vector data with no per-vector header.
- GT is stored in `.ibin` format as used in ANN benchmarks:
  - First `int32`: `Q` (number of queries)
  - Second `int32`: `K` (top-K neighbors per query)
  - Followed by `Q * K` `int32` neighbor IDs.

---

## 2. Adding Gaussian Noise to Replicated Vectors

### 2.1. Goal

Currently the dataset is replicated S times without changes:

```text
replica_shard_s[i] = original[i]
```

We want to change this to:

```text
replica_shard_s[i] = original[i] + ε_s[i]
```

where each element of `ε_s[i]` is drawn from a **Gaussian distribution**:

```text
ε_s[i][j] ~ Normal(0, noise^2)
```

This makes each replicated shard a **slightly jittered** version of the original vectors.

---

### 2.2. Batch-wise Processing

To keep memory usage under control, vectors are processed in batches.

- Let `arr` be a batch of vectors with shape `(B, D)`, where:
  - `B` is the batch size (e.g. `batch = 1024`).

#### Step 1 – Generate Gaussian Noise

```python
import numpy as np

noise_matrix = np.random.normal(
    loc=0.0,
    scale=noise,     # noise is a float, e.g. 1e-3
    size=arr.shape   # (B, D)
).astype(np.float32)
```

Notes:

- `loc=0.0` means the noise has mean 0.
- `scale=noise` is the standard deviation σ.
- `size=arr.shape` ensures noise is generated **per element**, not per vector.

#### Step 2 – Add Noise to the Batch

```python
noisy_batch = arr + noise_matrix
```

- `noisy_batch` remains shape `(B, D)`.
- Use the same dtype as the original vectors (`float32`).

#### Step 3 – Repeat Per Shard

For each shard index `s` in `0..S-1`, we must produce **a different noise matrix**:

```python
for s in range(S):
    noise_matrix = np.random.normal(0.0, noise, arr.shape).astype(np.float32)
    noisy_batch  = arr + noise_matrix
    # write noisy_batch to output for shard s
```

**Important nuances:**

1. **Independent noise per shard**  
   Each shard must have a new noise matrix — do *not* reuse the same jitter for all shards.
2. **Independent noise per element**  
   Noise is generated per element of the vector (B×D), not per vector or per batch scalar.
3. **Order of operations**  
   Noise is added **before** writing to disk.
4. **Type safety**  
   Ensure the summed result is cast to `float32` before writing.

---

## 3. New Dataset Layout After Replication + Noise

Original dataset IDs: `0 .. N-1`

After scaling by `S`, the new dataset is conceptually a concatenation of S shards:

```text
Shard 0: vectors 0      .. N-1
Shard 1: vectors N      .. 2N-1
Shard 2: vectors 2N     .. 3N-1
...
Shard s: vectors s*N    .. (s+1)*N - 1
...
Shard S-1: vectors (S-1)*N .. S*N - 1
```

For any original vector index `i` in `[0, N-1]`, its jittered replicas live at:

```text
i + 0*N,  i + 1*N,  i + 2*N,  ...  i + (S-1)*N
```

Each of these indices points to a vector that is **semantically the same** but **numerically jittered**.

---

## 4. Regenerating Ground-Truth (GT) for Jittered Replicas

### 4.1. Original GT

The original `.ibin` GT, once loaded, has shape `(Q, K)`:

```python
orig_gt[qid] = [id1, id2, ..., idK]
```

where each `idX` is an integer in `[0, N-1]` indexing into the **original dataset**.

### 4.2. Desired Semantics

Because each original vector now has S jittered replicas, **all S replicas are valid neighbors** for recall evaluation.

For each original neighbor `orig_id`, the correct set of neighbor IDs in the scaled dataset is:

```text
orig_id + 0*N, orig_id + 1*N, ..., orig_id + (S-1)*N
```

So we must expand each query’s neighbor list from length `K` to length `K * S`.

---

### 4.3. GT Expansion Algorithm

Let:

- `orig_gt` be a `(Q, K)` array of original neighbor IDs.
- `scaled_gt` be a `(Q, K*S)` array we are going to build.
- `S` be the replication factor.
- `N` be the original number of vectors.

Pseudocode in Python:

```python
Q, K = orig_gt.shape
scaled_gt = np.zeros((Q, K * S), dtype=np.int32)

for qid in range(Q):
    row = []
    for orig_id in orig_gt[qid]:
        for shard in range(S):
            new_id = orig_id + shard * N
            # Safety check
            if new_id >= N * S:
                raise ValueError("GT index out of range after scaling")
            row.append(new_id)
    scaled_gt[qid] = row
```

After this:

- `scaled_gt.shape == (Q, K * S)`
- `scaled_gt[qid]` contains all the valid jittered replicas for each original neighbor.

**Order of expanded neighbors:**

For a given query row with neighbors `[id1, id2, ..., idK]`, the expanded row is:

```text
[id1 + 0*N, id1 + 1*N, ..., id1 + (S-1)*N,
 id2 + 0*N, id2 + 1*N, ..., id2 + (S-1)*N,
 ...
 idK + 0*N, idK + 1*N, ..., idK + (S-1)*N]
```

Note:

- The *relative* ordering of neighbors does not matter for most recall evaluators — they only check if the returned ID is among the valid set.

---

### 4.4. Writing the New GT to .ibin

The new GT is saved in the same `.ibin` format:

1. First `int32`: `Q`  
2. Second `int32`: `K_new = K * S`  
3. Then `Q * K_new` `int32` neighbor IDs in row-major order.

Example write function:

```python
def write_gt_ibin(path, scaled_gt):
    Q, K_new = scaled_gt.shape
    header = np.array([Q, K_new], dtype=np.int32)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(scaled_gt.astype(np.int32).tobytes())
```

---

## 5. Sanity Checks and Edge Cases

### 5.1. Bounds Check

Ensure we never produce out-of-range indices:

```python
if new_id >= N * S:
    raise ValueError("GT index out of range after scaling")
```

### 5.2. Shape Validation

After building `scaled_gt`:

- Confirm `scaled_gt.shape == (Q, K * S)`.

### 5.3. Dataset Length Validation

After writing the scaled dataset, confirm the total number of vectors is `N * S`:

- For `.fvecs`: check file size vs. `((dim + 1) * 4)` bytes per vector.
- For `.fbin`: check file size vs. `(dim * 4)` bytes per vector.

### 5.4. Deterministic Noise (Optional)

If reproducibility is required, allow seeding:

```python
np.random.seed(seed_value)
```

This should be called **before** the first call to `np.random.normal`.

---

## 6. Summary (Checklist for Developer)

1. **Noise Injection**
   - For each batch `arr` (shape `(B, D)`), for each shard `s`:
     - Generate `noise_matrix = Normal(0, noise)` with `size=arr.shape`.
     - Compute `noisy_batch = arr + noise_matrix`.
     - Write `noisy_batch` as shard `s` to the output dataset.

2. **Dataset Layout**
   - Shard `s` occupies indices `[s * N .. (s + 1) * N - 1]`.

3. **GT Expansion**
   - For each original neighbor `orig_id` in `orig_gt[qid]`:
     - Generate IDs `orig_id + shard * N` for `shard in 0..S-1`.
   - Concatenate all such IDs to form `scaled_gt[qid]` of length `K * S`.

4. **GT File Format**
   - Write `Q`, then `K * S`, then `Q * K * S` neighbor IDs (`int32`) to `.ibin`.

5. **Validation**
   - Check `new_id < N * S`.
   - Check scaled dataset has `N * S` vectors.
   - Check `scaled_gt.shape == (Q, K * S)`.

Following these steps ensures the scaled dataset and the regenerated GT are fully consistent, and that all jittered copies are treated as valid neighbors in ANN recall evaluation.
