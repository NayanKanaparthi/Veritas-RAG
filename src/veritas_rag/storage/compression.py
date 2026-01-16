"""
Compression utilities for Veritas RAG.

MVP: zstd only via zstandard library.
"""

import zstandard as zstd


def compress_data(data: bytes, level: int = 3) -> bytes:
    """
    Compress data using zstd.

    Args:
        data: Data to compress
        level: Compression level (1-22, default 3)

    Returns:
        Compressed data
    """
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress_data(compressed_data: bytes) -> bytes:
    """
    Decompress data using zstd.

    Args:
        compressed_data: Compressed data

    Returns:
        Decompressed data
    """
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(compressed_data)
