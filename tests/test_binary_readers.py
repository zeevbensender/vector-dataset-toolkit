"""Tests for FBIN and IBIN readers."""

import struct
from pathlib import Path

import numpy as np
import pytest

from src.utils.io.fbin_reader import FBINReader, FBIN_HEADER_SIZE
from src.utils.io.ibin_reader import IBINReader, IBIN_HEADER_SIZE


def create_fbin_file(file_path: Path, data: np.ndarray) -> None:
    """Helper to create an FBIN file from NumPy data."""
    num_vectors, dimension = data.shape
    with open(file_path, "wb") as f:
        # Write header
        f.write(struct.pack("<II", num_vectors, dimension))
        # Write data
        f.write(data.astype(np.float32).tobytes())


def create_ibin_file(file_path: Path, data: np.ndarray) -> None:
    """Helper to create an IBIN file from NumPy data."""
    num_vectors, k = data.shape
    with open(file_path, "wb") as f:
        # Write header
        f.write(struct.pack("<II", num_vectors, k))
        # Write data
        f.write(data.astype(np.int32).tobytes())


class TestFBINReader:
    """Tests for FBINReader class."""

    def test_get_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction."""
        data = np.random.randn(100, 128).astype(np.float32)
        file_path = tmp_path / "test.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        metadata = reader.get_metadata()

        assert metadata["format"] == "fbin"
        assert metadata["vector_count"] == 100
        assert metadata["dimension"] == 128
        assert metadata["dtype"] == "float32"
        assert metadata["shape"] == (100, 128)
        assert metadata["size_match"] is True

    def test_sample(self, tmp_path: Path) -> None:
        """Test vector sampling."""
        data = np.random.randn(50, 64).astype(np.float32)
        file_path = tmp_path / "test_sample.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        sample = reader.sample(start=10, count=5)

        assert sample.shape == (5, 64)
        np.testing.assert_array_almost_equal(sample, data[10:15], decimal=5)

    def test_get_vector(self, tmp_path: Path) -> None:
        """Test single vector retrieval."""
        data = np.random.randn(30, 32).astype(np.float32)
        file_path = tmp_path / "test_get_vector.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        vector = reader.get_vector(15)

        np.testing.assert_array_almost_equal(vector, data[15], decimal=5)

    def test_read_sequential(self, tmp_path: Path) -> None:
        """Test sequential reading with chunks."""
        data = np.random.randn(100, 16).astype(np.float32)
        file_path = tmp_path / "test_sequential.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        
        chunks = list(reader.read_sequential(start=0, count=50, chunk_size=20))
        
        assert len(chunks) == 3  # 20 + 20 + 10
        assert chunks[0].shape == (20, 16)
        assert chunks[1].shape == (20, 16)
        assert chunks[2].shape == (10, 16)

    def test_len(self, tmp_path: Path) -> None:
        """Test __len__ method."""
        data = np.random.randn(75, 8).astype(np.float32)
        file_path = tmp_path / "test_len.fbin"
        create_fbin_file(file_path, data)

        reader = FBINReader(file_path)
        assert len(reader) == 75

    def test_file_not_found(self) -> None:
        """Test error handling for missing file."""
        reader = FBINReader("/nonexistent/path/file.fbin")
        with pytest.raises(FileNotFoundError):
            reader.get_metadata()


class TestIBINReader:
    """Tests for IBINReader class."""

    def test_get_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction."""
        data = np.random.randint(0, 1000, size=(100, 10)).astype(np.int32)
        file_path = tmp_path / "test.ibin"
        create_ibin_file(file_path, data)

        reader = IBINReader(file_path)
        metadata = reader.get_metadata()

        assert metadata["format"] == "ibin"
        assert metadata["vector_count"] == 100
        assert metadata["k"] == 10
        assert metadata["dtype"] == "int32"
        assert metadata["shape"] == (100, 10)
        assert metadata["size_match"] is True

    def test_sample(self, tmp_path: Path) -> None:
        """Test index sampling."""
        data = np.random.randint(0, 1000, size=(50, 20)).astype(np.int32)
        file_path = tmp_path / "test_sample.ibin"
        create_ibin_file(file_path, data)

        reader = IBINReader(file_path)
        sample = reader.sample(start=5, count=10)

        assert sample.shape == (10, 20)
        np.testing.assert_array_equal(sample, data[5:15])

    def test_get_neighbors(self, tmp_path: Path) -> None:
        """Test get_neighbors method."""
        data = np.random.randint(0, 1000, size=(30, 5)).astype(np.int32)
        file_path = tmp_path / "test_neighbors.ibin"
        create_ibin_file(file_path, data)

        reader = IBINReader(file_path)
        neighbors = reader.get_neighbors(10)

        np.testing.assert_array_equal(neighbors, data[10])

    def test_len(self, tmp_path: Path) -> None:
        """Test __len__ method."""
        data = np.random.randint(0, 100, size=(60, 5)).astype(np.int32)
        file_path = tmp_path / "test_len.ibin"
        create_ibin_file(file_path, data)

        reader = IBINReader(file_path)
        assert len(reader) == 60
