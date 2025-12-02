"""Tests for HDF5 reader."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.utils.io.hdf5_reader import HDF5Reader


class TestHDF5Reader:
    """Tests for HDF5Reader class."""

    def test_get_metadata_single_dataset(self, tmp_path: Path) -> None:
        """Test metadata extraction for file with single dataset."""
        data = np.random.randn(100, 128).astype(np.float32)
        file_path = tmp_path / "test_single.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=data)

        reader = HDF5Reader(file_path)
        metadata = reader.get_metadata()

        assert metadata["format"] == "hdf5"
        assert metadata["dataset_count"] == 1
        assert len(metadata["datasets"]) == 1
        assert metadata["datasets"][0]["path"] == "vectors"
        assert metadata["datasets"][0]["shape"] == (100, 128)

    def test_get_metadata_multiple_datasets(self, tmp_path: Path) -> None:
        """Test metadata extraction for file with multiple datasets."""
        file_path = tmp_path / "test_multi.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("train", data=np.random.randn(1000, 64).astype(np.float32))
            f.create_dataset("test", data=np.random.randn(100, 64).astype(np.float32))
            f.create_group("metadata")

        reader = HDF5Reader(file_path)
        metadata = reader.get_metadata()

        assert metadata["dataset_count"] == 2
        assert len(metadata["groups"]) == 1

    def test_get_dataset_metadata(self, tmp_path: Path) -> None:
        """Test metadata extraction for a specific dataset."""
        data = np.random.randn(50, 32).astype(np.float32)
        file_path = tmp_path / "test_ds_meta.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=data, compression="gzip")

        reader = HDF5Reader(file_path)
        metadata = reader.get_metadata("vectors")

        assert metadata["dataset_path"] == "vectors"
        assert metadata["shape"] == (50, 32)
        assert metadata["dtype"] == "float32"
        assert metadata["vector_count"] == 50
        assert metadata["dimension"] == 32
        assert metadata["compression"] == "gzip"

    def test_list_contents(self, tmp_path: Path) -> None:
        """Test listing groups and datasets."""
        file_path = tmp_path / "test_contents.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_group("group1")
            f.create_group("group1/subgroup")
            f.create_dataset("dataset1", data=np.zeros(10))
            f.create_dataset("group1/dataset2", data=np.zeros(10))

        reader = HDF5Reader(file_path)
        contents = reader.list_contents()

        assert "group1" in contents["groups"]
        assert "group1/subgroup" in contents["groups"]
        assert "dataset1" in contents["datasets"]
        assert "group1/dataset2" in contents["datasets"]

    def test_sample(self, tmp_path: Path) -> None:
        """Test vector sampling."""
        data = np.random.randn(100, 64).astype(np.float32)
        file_path = tmp_path / "test_sample.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=data)

        reader = HDF5Reader(file_path)
        sample = reader.sample("vectors", start=10, count=5)

        assert sample.shape == (5, 64)
        np.testing.assert_array_equal(sample, data[10:15])

    def test_get_vector(self, tmp_path: Path) -> None:
        """Test single vector retrieval."""
        data = np.random.randn(50, 32).astype(np.float32)
        file_path = tmp_path / "test_get_vector.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=data)

        reader = HDF5Reader(file_path)
        vector = reader.get_vector("vectors", 25)

        np.testing.assert_array_equal(vector, data[25])

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        data = np.random.randn(10, 8).astype(np.float32)
        file_path = tmp_path / "test_context.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=data)

        with HDF5Reader(file_path) as reader:
            metadata = reader.get_metadata()
            assert metadata["dataset_count"] == 1

    def test_file_not_found(self) -> None:
        """Test error handling for missing file."""
        reader = HDF5Reader("/nonexistent/path/file.h5")
        with pytest.raises(FileNotFoundError):
            reader.get_metadata()

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        """Test error handling for missing dataset."""
        file_path = tmp_path / "test_no_ds.h5"
        
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset("vectors", data=np.zeros(10))

        reader = HDF5Reader(file_path)
        with pytest.raises(KeyError):
            reader.get_metadata("nonexistent")
