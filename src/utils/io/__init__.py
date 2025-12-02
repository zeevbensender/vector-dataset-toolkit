"""I/O utilities for reading and writing vector dataset files."""

from .npy_reader import NPYReader
from .hdf5_reader import HDF5Reader
from .hdf5_unwrapper import HDF5Unwrapper
from .fbin_reader import FBINReader, FBIN_HEADER_SIZE
from .ibin_reader import IBINReader, IBIN_HEADER_SIZE
from .ibin_writer import IBINWriter, write_ibin
from .fbin_writer import FBINWriter, write_fbin
from .converter import Converter
from .fbin_converter import (
    FBINConverter,
    convert_fbin_to_npy,
    convert_npy_to_fbin,
    convert_fbin_to_hdf5,
)
from .hdf5_wrapper import HDF5Wrapper
from .shard_merger import (
    ShardMerger,
    ShardInfo,
    ShardValidationResult,
    MergePreview,
    merge_fbin_shards,
)

__all__ = [
    "NPYReader",
    "HDF5Reader",
    "HDF5Unwrapper",
    "FBINReader",
    "FBIN_HEADER_SIZE",
    "IBINReader",
    "IBIN_HEADER_SIZE",
    "IBINWriter",
    "write_ibin",
    "FBINWriter",
    "write_fbin",
    "Converter",
    "FBINConverter",
    "convert_fbin_to_npy",
    "convert_npy_to_fbin",
    "convert_fbin_to_hdf5",
    "HDF5Wrapper",
    "ShardMerger",
    "ShardInfo",
    "ShardValidationResult",
    "MergePreview",
    "merge_fbin_shards",
]
