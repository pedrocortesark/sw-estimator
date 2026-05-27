"""Loaders sub-package — physical access to file sources.

A loader knows *how* to reach a file (local disk, HTTP, Drive, S3, …) but
knows nothing about its content.  It always returns raw ``bytes``.

Public interface::

    from src.ingest.loaders import filesystem, http, drive
    raw: bytes = filesystem.load("/data/budgets/2024-q1.json")
    raw: bytes = http.load("https://example.com/contract.pdf")
"""
