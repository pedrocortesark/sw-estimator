"""Parsers sub-package — format-specific text extraction.

Each parser module knows how to read one file format and produces an
*intermediate representation* (IR) that is specific to that format.  The
normalizer layer converts the IR to canonical :class:`~src.ingest.models.Document`
instances.

Intermediate representations by format
---------------------------------------
+---------------+-----------------------------+----------------------------+
| Parser        | IR type                     | Granularity                |
+===============+=============================+============================+
| json_parser   | ``list[ParsedBlock]``       | one block per array item   |
+---------------+-----------------------------+----------------------------+
| txt_parser    | ``list[Turn]``              | one turn per speaker turn  |
+---------------+-----------------------------+----------------------------+
| xlsx_parser   | ``list[ParsedTable]``       | one table per worksheet    |
+---------------+-----------------------------+----------------------------+
| docx_parser   | ``list[ParsedSection]``     | one section per heading    |
+---------------+-----------------------------+----------------------------+
| pdf_parser    | ``list[ParsedPage]``        | one page per PDF page      |
+---------------+-----------------------------+----------------------------+

The three-layer design (loader → parser → normalizer) is intentional.
Parsers are tested against their own IR; normalizers are tested against the
canonical contract.  Each test stays focused without needing to stub the other
layer.
"""
