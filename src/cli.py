"""Command-line interface for Vector Dataset Toolkit.

This module provides headless CLI access to the conversion functionality.

Usage:
    python -m src.cli convert input.npy output.h5
    python -m src.cli convert input.h5 output.npy --dataset vectors
    python -m src.cli info file.npy
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from .utils.io import (
    Converter,
    FBINReader,
    HDF5Reader,
    IBINReader,
    NPYReader,
)


def print_progress(current: int, total: int) -> None:
    """Print progress to stdout.
    
    Args:
        current: Current progress value.
        total: Total progress value.
    """
    if total > 0:
        percent = int((current / total) * 100)
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {percent}% ({current:,}/{total:,})", end="", flush=True)


def cmd_info(args: argparse.Namespace) -> int:
    """Display file information.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success).
    """
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".npy":
            reader = NPYReader(file_path)
            metadata = reader.get_metadata()
        elif suffix in (".h5", ".hdf5"):
            reader = HDF5Reader(file_path)
            metadata = reader.get_metadata()
        elif suffix == ".fbin":
            reader = FBINReader(file_path)
            metadata = reader.get_metadata()
        elif suffix == ".ibin":
            reader = IBINReader(file_path)
            metadata = reader.get_metadata()
        else:
            print(f"Error: Unsupported format: {suffix}", file=sys.stderr)
            return 1

        print(f"\nFile: {file_path}")
        print("-" * 50)
        for key, value in metadata.items():
            if key == "datasets" and isinstance(value, list):
                print(f"  {key}:")
                for ds in value:
                    print(f"    - {ds.get('path', 'N/A')}: {ds.get('shape', 'N/A')}")
            else:
                print(f"  {key}: {value}")
        print()
        return 0

    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert a file to another format.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success).
    """
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    input_suffix = input_path.suffix.lower()
    output_suffix = output_path.suffix.lower()

    # Validate conversion is supported
    supported = [
        (".npy", ".h5"),
        (".npy", ".hdf5"),
        (".h5", ".npy"),
        (".hdf5", ".npy"),
    ]
    
    if (input_suffix, output_suffix) not in supported:
        print(
            f"Error: Unsupported conversion: {input_suffix} -> {output_suffix}",
            file=sys.stderr
        )
        print("Supported conversions: NPY <-> HDF5", file=sys.stderr)
        return 1

    try:
        converter = Converter(
            chunk_size=args.chunk_size,
            progress_callback=print_progress if not args.quiet else None,
        )

        print(f"Converting: {input_path} -> {output_path}")
        
        if input_suffix == ".npy":
            compression = args.compression if args.compression != "none" else None
            result = converter.npy_to_hdf5(
                input_path,
                output_path,
                dataset_name=args.dataset,
                compression=compression,
            )
        else:
            result = converter.hdf5_to_npy(
                input_path,
                output_path,
                dataset_path=args.dataset,
            )

        if not args.quiet:
            print()  # Newline after progress bar
        print(f"\nConversion complete!")
        print(f"  Vectors converted: {result.get('vectors_converted', 'N/A'):,}")
        print(f"  Shape: {result.get('shape', 'N/A')}")
        print(f"  Output: {output_path}")
        
        return 0

    except Exception as e:
        print(f"\nError during conversion: {e}", file=sys.stderr)
        return 1


def cmd_sample(args: argparse.Namespace) -> int:
    """Sample vectors from a file.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success).
    """
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".npy":
            reader = NPYReader(file_path)
            sample = reader.sample(args.start, args.count)
        elif suffix in (".h5", ".hdf5"):
            reader = HDF5Reader(file_path)
            contents = reader.list_contents()
            ds_path = args.dataset or (contents["datasets"][0] if contents["datasets"] else None)
            if not ds_path:
                print("Error: No datasets found in HDF5 file", file=sys.stderr)
                return 1
            sample = reader.sample(ds_path, args.start, args.count)
        elif suffix == ".fbin":
            reader = FBINReader(file_path)
            sample = reader.sample(args.start, args.count)
        elif suffix == ".ibin":
            reader = IBINReader(file_path)
            sample = reader.sample(args.start, args.count)
        else:
            print(f"Error: Unsupported format: {suffix}", file=sys.stderr)
            return 1

        print(f"\nSample from {file_path} (indices {args.start}-{args.start + len(sample) - 1}):")
        print("-" * 50)
        for i, vector in enumerate(sample):
            idx = args.start + i
            if len(vector) > 10:
                vec_str = f"[{', '.join(f'{v:.4f}' for v in vector[:5])} ... {', '.join(f'{v:.4f}' for v in vector[-3:])}]"
            else:
                vec_str = f"[{', '.join(f'{v:.4f}' for v in vector)}]"
            print(f"[{idx}]: {vec_str}")
        
        return 0

    except Exception as e:
        print(f"Error sampling file: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="vdt-cli",
        description="Vector Dataset Toolkit - Command Line Interface"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display file information")
    info_parser.add_argument("file", help="Path to the vector dataset file")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between formats")
    convert_parser.add_argument("input", help="Input file path")
    convert_parser.add_argument("output", help="Output file path")
    convert_parser.add_argument(
        "-d", "--dataset",
        default="vectors",
        help="Dataset name (for HDF5 files, default: vectors)"
    )
    convert_parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for processing (default: 10000)"
    )
    convert_parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression for HDF5 output (default: gzip)"
    )
    convert_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample vectors from a file")
    sample_parser.add_argument("file", help="Path to the vector dataset file")
    sample_parser.add_argument(
        "-s", "--start",
        type=int,
        default=0,
        help="Starting index (default: 0)"
    )
    sample_parser.add_argument(
        "-n", "--count",
        type=int,
        default=10,
        help="Number of vectors to sample (default: 10)"
    )
    sample_parser.add_argument(
        "-d", "--dataset",
        help="Dataset path (for HDF5 files)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        return cmd_info(args)
    elif args.command == "convert":
        return cmd_convert(args)
    elif args.command == "sample":
        return cmd_sample(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
