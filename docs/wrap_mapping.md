# HDF5 <-> FBIN/IBIN Mapping

The Inspector screenshot shows a wrapped HDF5 with five datasets:

- `base` — full vector corpus used for ANN search (100k × 1536 in the example)
- `train` — an alias to the same vectors as `base`
- `test` — query vectors (5k × 1536)
- `neighbors` — ground-truth neighbor indices for each query (5k × k)
- `distances` — optional per-query distances that are **not** present in the FBIN/IBIN trio

When such a file is extracted to the flat binaries described elsewhere in this repo:

- `base.fbin` corresponds to `base` (and therefore also `train`, since `train` links to the same data)
- `queries.fbin` corresponds to `test`
- `gt.ibin` corresponds to `neighbors`
- `distances` has no FBIN/IBIN counterpart and is omitted on export

The Wrap flow mirrors this mapping: base shards feed the `base` dataset and a `train` alias, the queries FBIN feeds `test`, and an optional IBIN feeds `neighbors`.
