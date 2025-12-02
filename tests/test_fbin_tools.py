"""Tests for FBIN writer, converter, and shard merger."""

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.utils.io.fbin_reader import FBINReader, FBIN_HEADER_SIZE
from src.utils.io.fbin_writer import FBINWriter, write_fbin
from src.utils.io.fbin_converter import (
    FBINConverter,
    convert_fbin_to_npy,
    convert_npy_to_fbin,
    convert_fbin_to_hdf5,
)
from src.utils.io.shard_merger import (
    ShardMerger,
    ShardValidationResult,
    merge_fbin_shards,
)


def create_fbin_file(file_path: Path, data: np.ndarray) -> None:
    """Helper to create an FBIN file from NumPy data."""
    num_vectors, dimension = data.shape
    with open(file_path, "wb") as f:
        f.write(struct.pack("<II", num_vectors, dimension))
        f.write(data.astype(np.float32).tobytes())


class TestFBINWriter:
    """Tests for FBINWriter class."""

    def test_write_basic(self, tmp_path: Path) -> None:
        """Test basic FBIN writing."""
        data = np.random.randn(100, 64).astype(np.float32)
        output_path = tmp_path / "output.fbin"

        writer = FBINWriter(output_path)
        result = writer.write(data)

        assert result["vector_count"] == 100
        assert result["dimension"] == 64
        assert output_path.exists()

        # Verify written data
        reader = FBINReader(output_path)
        loaded = reader.load()
        np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_write_with_checksum(self, tmp_path: Path) -> None:
        """Test writing with checksum computation."""
        data = np.random.randn(50, 32).astype(np.float32)
        output_path = tmp_path / "output.fbin"

        writer = FBINWriter(output_path)
        result = writer.write(data, compute_checksum=True)

        assert "checksum" in result
        assert len(result["checksum"]) == 64  # SHA256 hex digest

    def test_write_progress_callback(self, tmp_path: Path) -> None:
        """Test progress callback during write."""
        data = np.random.randn(100, 16).astype(np.float32)
        output_path = tmp_path / "output.fbin"

        progress_calls = []
        def on_progress(current: int, total: int):
            progress_calls.append((current, total))

        writer = FBINWriter(output_path, progress_callback=on_progress)
        writer.write(data, chunk_size=25)

        assert len(progress_calls) == 4
        assert progress_calls[-1] == (100, 100)

    def test_write_creates_directory(self, tmp_path: Path) -> None:
        """Test that write creates output directory if needed."""
        data = np.random.randn(20, 8).astype(np.float32)
        output_path = tmp_path / "subdir" / "output.fbin"

        writer = FBINWriter(output_path)
        writer.write(data)

        assert output_path.exists()

    def test_write_dtype_conversion(self, tmp_path: Path) -> None:
        """Test that non-float32 data is converted."""
        data = np.random.randn(30, 16).astype(np.float64)
        output_path = tmp_path / "output.fbin"

        writer = FBINWriter(output_path)
        writer.write(data)

        reader = FBINReader(output_path)
        metadata = reader.get_metadata()
        assert metadata["dtype"] == "float32"

    def test_write_from_chunks(self, tmp_path: Path) -> None:
        """Test writing from an iterable of chunks."""
        chunks = [np.random.randn(20, 32).astype(np.float32) for _ in range(5)]
        output_path = tmp_path / "output.fbin"

        writer = FBINWriter(output_path)
        result = writer.write_from_chunks(
            iter(chunks),
            num_vectors=100,
            dimension=32,
        )

        assert result["vector_count"] == 100
        assert result["dimension"] == 32

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test write_fbin convenience function."""
        data = np.random.randn(50, 24).astype(np.float32)
        output_path = tmp_path / "output.fbin"

        result = write_fbin(output_path, data, compute_checksum=True)

        assert result["vector_count"] == 50
        assert "checksum" in result


class TestFBINConverter:
    """Tests for FBINConverter class."""

    def test_fbin_to_npy(self, tmp_path: Path) -> None:
        """Test FBIN to NPY conversion."""
        data = np.random.randn(100, 64).astype(np.float32)
        input_path = tmp_path / "input.fbin"
        output_path = tmp_path / "output.npy"
        create_fbin_file(input_path, data)

        converter = FBINConverter(chunk_size=25)
        result = converter.fbin_to_npy(input_path, output_path)

        assert result["vector_count"] == 100
        assert result["dimension"] == 64
        assert output_path.exists()

        # Verify data
        loaded = np.load(str(output_path))
        np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_npy_to_fbin(self, tmp_path: Path) -> None:
        """Test NPY to FBIN conversion."""
        data = np.random.randn(80, 32).astype(np.float32)
        input_path = tmp_path / "input.npy"
        output_path = tmp_path / "output.fbin"
        np.save(str(input_path), data)

        converter = FBINConverter(chunk_size=20)
        result = converter.npy_to_fbin(input_path, output_path)

        assert result["vector_count"] == 80
        assert output_path.exists()

        # Verify data
        reader = FBINReader(output_path)
        loaded = reader.load()
        np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_fbin_to_hdf5(self, tmp_path: Path) -> None:
        """Test FBIN to HDF5 conversion."""
        data = np.random.randn(60, 48).astype(np.float32)
        input_path = tmp_path / "input.fbin"
        output_path = tmp_path / "output.h5"
        create_fbin_file(input_path, data)

        converter = FBINConverter()
        result = converter.fbin_to_hdf5(input_path, output_path)

        assert result["vector_count"] == 60
        assert output_path.exists()

        # Verify data
        with h5py.File(str(output_path), "r") as f:
            loaded = f["vectors"][:]
            np.testing.assert_array_almost_equal(loaded, data, decimal=5)

    def test_dry_run_fbin_to_npy(self, tmp_path: Path) -> None:
        """Test dry run mode."""
        data = np.random.randn(50, 32).astype(np.float32)
        input_path = tmp_path / "input.fbin"
        output_path = tmp_path / "output.npy"
        create_fbin_file(input_path, data)

        converter = FBINConverter()
        result = converter.fbin_to_npy(input_path, output_path, dry_run=True)

        assert result["dry_run"] is True
        assert result["vector_count"] == 50
        assert not output_path.exists()

    def test_roundtrip_fbin_npy_fbin(self, tmp_path: Path) -> None:
        """Test roundtrip conversion FBIN -> NPY -> FBIN."""
        original_data = np.random.randn(40, 24).astype(np.float32)
        original_fbin = tmp_path / "original.fbin"
        npy_path = tmp_path / "intermediate.npy"
        final_fbin = tmp_path / "final.fbin"
        create_fbin_file(original_fbin, original_data)

        converter = FBINConverter()
        converter.fbin_to_npy(original_fbin, npy_path)
        converter.npy_to_fbin(npy_path, final_fbin)

        # Verify roundtrip
        reader = FBINReader(final_fbin)
        loaded = reader.load()
        np.testing.assert_array_almost_equal(loaded, original_data, decimal=5)

    def test_convenience_functions(self, tmp_path: Path) -> None:
        """Test convenience conversion functions."""
        data = np.random.randn(30, 16).astype(np.float32)
        fbin_path = tmp_path / "test.fbin"
        npy_path = tmp_path / "test.npy"
        h5_path = tmp_path / "test.h5"
        create_fbin_file(fbin_path, data)

        # FBIN -> NPY
        result = convert_fbin_to_npy(fbin_path, npy_path)
        assert result["vector_count"] == 30

        # NPY -> FBIN (roundtrip)
        fbin2_path = tmp_path / "test2.fbin"
        result = convert_npy_to_fbin(npy_path, fbin2_path)
        assert result["vector_count"] == 30

        # FBIN -> HDF5
        result = convert_fbin_to_hdf5(fbin_path, h5_path)
        assert result["vector_count"] == 30


class TestShardMerger:
    """Tests for ShardMerger class."""

    def test_validate_single_shard(self, tmp_path: Path) -> None:
        """Test single shard validation."""
        data = np.random.randn(50, 32).astype(np.float32)
        shard_path = tmp_path / "shard.fbin"
        create_fbin_file(shard_path, data)

        merger = ShardMerger()
        info = merger.validate_shard(shard_path)

        assert info.validation_result == ShardValidationResult.COMPATIBLE
        assert info.vector_count == 50
        assert info.dimension == 32

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        """Test validation of missing file."""
        merger = ShardMerger()
        info = merger.validate_shard(tmp_path / "nonexistent.fbin")

        assert info.validation_result == ShardValidationResult.FILE_NOT_FOUND

    def test_validate_dimension_mismatch(self, tmp_path: Path) -> None:
        """Test validation with dimension mismatch."""
        data = np.random.randn(50, 32).astype(np.float32)
        shard_path = tmp_path / "shard.fbin"
        create_fbin_file(shard_path, data)

        merger = ShardMerger()
        info = merger.validate_shard(shard_path, reference_dimension=64)

        assert info.validation_result == ShardValidationResult.INCOMPATIBLE_DIMENSION

    def test_validate_multiple_shards(self, tmp_path: Path) -> None:
        """Test validation of multiple shards."""
        # Create compatible shards
        for i in range(3):
            data = np.random.randn(40, 24).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(3)]

        merger = ShardMerger()
        infos = merger.validate_shards(shard_paths)

        assert len(infos) == 3
        assert all(
            info.validation_result == ShardValidationResult.COMPATIBLE
            for info in infos
        )

    def test_preview_merge(self, tmp_path: Path) -> None:
        """Test merge preview."""
        # Create shards
        for i in range(3):
            data = np.random.randn(50, 32).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(3)]

        merger = ShardMerger()
        preview = merger.preview_merge(shard_paths)

        assert preview.total_vectors == 150
        assert preview.dimension == 32
        assert preview.all_compatible is True
        assert len(preview.incompatible_shards) == 0

    def test_merge_to_fbin(self, tmp_path: Path) -> None:
        """Test merging shards to FBIN format."""
        # Create shards with known data
        all_data = []
        for i in range(3):
            data = np.random.randn(40, 24).astype(np.float32) * (i + 1)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)
            all_data.append(data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(3)]
        output_path = tmp_path / "merged.fbin"

        merger = ShardMerger(chunk_size=20)
        result = merger.merge(shard_paths, output_path, output_format="fbin")

        assert result["total_vectors"] == 120
        assert result["shards_merged"] == 3
        assert output_path.exists()

        # Verify merged data
        reader = FBINReader(output_path)
        merged = reader.load()
        expected = np.vstack(all_data)
        np.testing.assert_array_almost_equal(merged, expected, decimal=5)

    def test_merge_to_npy(self, tmp_path: Path) -> None:
        """Test merging shards to NPY format."""
        # Create shards
        all_data = []
        for i in range(2):
            data = np.random.randn(30, 16).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)
            all_data.append(data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(2)]
        output_path = tmp_path / "merged.npy"

        merger = ShardMerger()
        result = merger.merge(shard_paths, output_path, output_format="npy")

        assert result["total_vectors"] == 60
        assert output_path.exists()

        # Verify merged data
        merged = np.load(str(output_path))
        expected = np.vstack(all_data)
        np.testing.assert_array_almost_equal(merged, expected, decimal=5)

    def test_merge_with_checksum(self, tmp_path: Path) -> None:
        """Test merge with checksum computation."""
        for i in range(2):
            data = np.random.randn(25, 16).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(2)]
        output_path = tmp_path / "merged.fbin"

        merger = ShardMerger()
        result = merger.merge(
            shard_paths, output_path,
            compute_checksum=True
        )

        assert "checksum" in result
        assert len(result["checksum"]) == 64

    def test_merge_dry_run(self, tmp_path: Path) -> None:
        """Test merge dry run mode."""
        for i in range(2):
            data = np.random.randn(30, 24).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(2)]
        output_path = tmp_path / "merged.fbin"

        merger = ShardMerger()
        result = merger.merge(shard_paths, output_path, dry_run=True)

        assert result["dry_run"] is True
        assert result["total_vectors"] == 60
        assert not output_path.exists()

    def test_merge_incompatible_shards_fails(self, tmp_path: Path) -> None:
        """Test that merging incompatible shards raises error."""
        # Create shards with different dimensions
        data1 = np.random.randn(30, 32).astype(np.float32)
        data2 = np.random.randn(30, 64).astype(np.float32)  # Different dimension
        create_fbin_file(tmp_path / "shard_0.fbin", data1)
        create_fbin_file(tmp_path / "shard_1.fbin", data2)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(2)]
        output_path = tmp_path / "merged.fbin"

        merger = ShardMerger()
        with pytest.raises(ValueError, match="incompatible"):
            merger.merge(shard_paths, output_path)

    def test_convenience_function(self, tmp_path: Path) -> None:
        """Test merge_fbin_shards convenience function."""
        for i in range(2):
            data = np.random.randn(20, 16).astype(np.float32)
            create_fbin_file(tmp_path / f"shard_{i}.fbin", data)

        shard_paths = [tmp_path / f"shard_{i}.fbin" for i in range(2)]
        output_path = tmp_path / "merged.fbin"

        result = merge_fbin_shards(shard_paths, output_path)

        assert result["total_vectors"] == 40


class TestFBINReaderSampling:
    """Tests for new sampling methods in FBINReader."""

    def test_sample_random(self, tmp_path: Path) -> None:
        """Test random sampling."""
        data = np.random.randn(100, 32).astype(np.float32)
        file_path = tmp_path / "test.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        indices, vectors = reader.sample_random(count=10, seed=42)

        assert len(indices) == 10
        assert vectors.shape == (10, 32)
        # Verify vectors match original data at those indices
        np.testing.assert_array_almost_equal(vectors, data[indices], decimal=5)

    def test_sample_random_reproducible(self, tmp_path: Path) -> None:
        """Test that random sampling with same seed is reproducible."""
        data = np.random.randn(100, 32).astype(np.float32)
        file_path = tmp_path / "test.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        idx1, vec1 = reader.sample_random(count=5, seed=123)
        idx2, vec2 = reader.sample_random(count=5, seed=123)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(vec1, vec2)

    def test_sample_strided(self, tmp_path: Path) -> None:
        """Test strided sampling."""
        data = np.random.randn(100, 16).astype(np.float32)
        file_path = tmp_path / "test.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        vectors = reader.sample_strided(stride=10)

        assert vectors.shape == (10, 16)  # 0, 10, 20, ..., 90
        np.testing.assert_array_almost_equal(vectors, data[::10], decimal=5)

    def test_sample_strided_with_max(self, tmp_path: Path) -> None:
        """Test strided sampling with max count."""
        data = np.random.randn(100, 16).astype(np.float32)
        file_path = tmp_path / "test.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        vectors = reader.sample_strided(stride=10, max_count=5)

        assert vectors.shape == (5, 16)
