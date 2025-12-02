"""Tests for the conversion core."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.utils.io.converter import Converter, convert_npy_to_hdf5, convert_hdf5_to_npy


class TestConverter:
    """Tests for Converter class."""

    def test_npy_to_hdf5(self, tmp_path: Path) -> None:
        """Test NPY to HDF5 conversion."""
        # Create input NPY file
        data = np.random.randn(100, 64).astype(np.float32)
        input_path = tmp_path / "input.npy"
        np.save(str(input_path), data)

        output_path = tmp_path / "output.h5"

        # Convert
        converter = Converter(chunk_size=25)
        result = converter.npy_to_hdf5(input_path, output_path)

        # Verify result
        assert result["vectors_converted"] == 100
        assert result["shape"] == (100, 64)

        # Verify output file
        with h5py.File(str(output_path), "r") as f:
            assert "vectors" in f
            output_data = f["vectors"][:]
            np.testing.assert_array_equal(output_data, data)

    def test_hdf5_to_npy(self, tmp_path: Path) -> None:
        """Test HDF5 to NPY conversion."""
        # Create input HDF5 file
        data = np.random.randn(80, 32).astype(np.float32)
        input_path = tmp_path / "input.h5"
        with h5py.File(str(input_path), "w") as f:
            f.create_dataset("vectors", data=data)

        output_path = tmp_path / "output.npy"

        # Convert
        converter = Converter(chunk_size=20)
        result = converter.hdf5_to_npy(input_path, output_path)

        # Verify result
        assert result["vectors_converted"] == 80

        # Verify output file
        output_data = np.load(str(output_path))
        np.testing.assert_array_equal(output_data, data)

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Test progress callback is called."""
        data = np.random.randn(50, 16).astype(np.float32)
        input_path = tmp_path / "input.npy"
        np.save(str(input_path), data)

        output_path = tmp_path / "output.h5"

        progress_calls = []
        def on_progress(current: int, total: int):
            progress_calls.append((current, total))

        converter = Converter(chunk_size=10, progress_callback=on_progress)
        converter.npy_to_hdf5(input_path, output_path)

        # Should have 5 progress calls (50 vectors / 10 chunk size)
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (50, 50)

    def test_compression(self, tmp_path: Path) -> None:
        """Test HDF5 compression option."""
        data = np.random.randn(100, 32).astype(np.float32)
        input_path = tmp_path / "input.npy"
        np.save(str(input_path), data)

        output_path = tmp_path / "output.h5"

        converter = Converter()
        converter.npy_to_hdf5(
            input_path, output_path,
            compression="gzip", compression_opts=9
        )

        with h5py.File(str(output_path), "r") as f:
            assert f["vectors"].compression == "gzip"

    def test_custom_dataset_name(self, tmp_path: Path) -> None:
        """Test custom dataset name in HDF5."""
        data = np.random.randn(20, 8).astype(np.float32)
        input_path = tmp_path / "input.npy"
        np.save(str(input_path), data)

        output_path = tmp_path / "output.h5"

        converter = Converter()
        converter.npy_to_hdf5(input_path, output_path, dataset_name="embeddings")

        with h5py.File(str(output_path), "r") as f:
            assert "embeddings" in f
            assert "vectors" not in f

    def test_convenience_functions(self, tmp_path: Path) -> None:
        """Test convenience functions."""
        # Test convert_npy_to_hdf5
        data = np.random.randn(30, 16).astype(np.float32)
        npy_path = tmp_path / "data.npy"
        np.save(str(npy_path), data)

        h5_path = tmp_path / "data.h5"
        result = convert_npy_to_hdf5(npy_path, h5_path)
        assert result["vectors_converted"] == 30

        # Test convert_hdf5_to_npy
        npy_path2 = tmp_path / "data_back.npy"
        result = convert_hdf5_to_npy(h5_path, npy_path2)
        assert result["vectors_converted"] == 30

        # Verify roundtrip
        roundtrip_data = np.load(str(npy_path2))
        np.testing.assert_array_equal(roundtrip_data, data)

    def test_cancellation(self, tmp_path: Path) -> None:
        """Test conversion cancellation."""
        data = np.random.randn(100, 16).astype(np.float32)
        input_path = tmp_path / "input.npy"
        np.save(str(input_path), data)

        output_path = tmp_path / "output.h5"

        def cancel_on_progress(current: int, total: int):
            if current >= 20:
                converter.cancel()

        converter = Converter(chunk_size=10, progress_callback=cancel_on_progress)
        
        with pytest.raises(RuntimeError, match="cancelled"):
            converter.npy_to_hdf5(input_path, output_path)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error handling for missing input file."""
        converter = Converter()
        
        with pytest.raises(FileNotFoundError):
            converter.npy_to_hdf5(
                tmp_path / "nonexistent.npy",
                tmp_path / "output.h5"
            )

    def test_hdf5_dataset_not_found(self, tmp_path: Path) -> None:
        """Test error handling for missing dataset in HDF5."""
        input_path = tmp_path / "input.h5"
        with h5py.File(str(input_path), "w") as f:
            f.create_dataset("data", data=np.zeros(10))

        converter = Converter()
        
        with pytest.raises(KeyError):
            converter.hdf5_to_npy(
                input_path,
                tmp_path / "output.npy",
                dataset_path="nonexistent"
            )
