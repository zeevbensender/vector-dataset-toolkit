"""Tests for NPY reader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.io.npy_reader import NPYReader


class TestNPYReader:
    """Tests for NPYReader class."""

    def test_get_metadata_2d_array(self, tmp_path: Path) -> None:
        """Test metadata extraction for 2D array."""
        # Create test data
        data = np.random.randn(100, 128).astype(np.float32)
        file_path = tmp_path / "test_vectors.npy"
        np.save(str(file_path), data)

        # Read metadata
        reader = NPYReader(file_path)
        metadata = reader.get_metadata()

        assert metadata["format"] == "npy"
        assert metadata["shape"] == (100, 128)
        assert metadata["dtype"] == "float32"
        assert metadata["vector_count"] == 100
        assert metadata["dimension"] == 128
        assert metadata["ndim"] == 2

    def test_get_metadata_1d_array(self, tmp_path: Path) -> None:
        """Test metadata extraction for 1D array."""
        data = np.arange(50).astype(np.int32)
        file_path = tmp_path / "test_1d.npy"
        np.save(str(file_path), data)

        reader = NPYReader(file_path)
        metadata = reader.get_metadata()

        assert metadata["shape"] == (50,)
        assert metadata["dtype"] == "int32"
        assert metadata["vector_count"] == 50

    def test_sample(self, tmp_path: Path) -> None:
        """Test vector sampling."""
        data = np.random.randn(100, 64).astype(np.float32)
        file_path = tmp_path / "test_sample.npy"
        np.save(str(file_path), data)

        reader = NPYReader(file_path)
        sample = reader.sample(start=10, count=5)

        assert sample.shape == (5, 64)
        np.testing.assert_array_equal(sample, data[10:15])

    def test_get_vector(self, tmp_path: Path) -> None:
        """Test single vector retrieval."""
        data = np.random.randn(50, 32).astype(np.float32)
        file_path = tmp_path / "test_get_vector.npy"
        np.save(str(file_path), data)

        reader = NPYReader(file_path)
        vector = reader.get_vector(25)

        np.testing.assert_array_equal(vector, data[25])

    def test_len(self, tmp_path: Path) -> None:
        """Test __len__ method."""
        data = np.random.randn(75, 16).astype(np.float32)
        file_path = tmp_path / "test_len.npy"
        np.save(str(file_path), data)

        reader = NPYReader(file_path)
        assert len(reader) == 75

    def test_file_not_found(self) -> None:
        """Test error handling for missing file."""
        reader = NPYReader("/nonexistent/path/file.npy")
        with pytest.raises(FileNotFoundError):
            reader.get_metadata()

    def test_mmap_mode_none(self, tmp_path: Path) -> None:
        """Test loading without memory mapping."""
        data = np.random.randn(20, 8).astype(np.float32)
        file_path = tmp_path / "test_no_mmap.npy"
        np.save(str(file_path), data)

        reader = NPYReader(file_path, mmap_mode=None)
        loaded = reader.load()

        np.testing.assert_array_equal(loaded, data)
