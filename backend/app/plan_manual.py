"""Local extraction of a plan reference document into ``MaterialInput`` (``POST /api/plan/manual``).

The user hands the server raw bytes; this module turns them into the bounded ``MaterialInput`` a plan
start can carry as one of its ``materials``. Nothing here calls a model, reaches the network, or writes to the filesystem: the only side
effect is a validated value returned to the route, so an import needs no AI consent (there is no AI step).
A file either fits whole or is refused, never truncated, whatever fails is one of the closed-set codes the
frontend maps (``manual_too_large``, ``manual_too_long``, ``manual_empty``, ``manual_invalid``,
``manual_unsupported``).

Supported, and how each is read:

* ``.txt`` / ``.md`` — strict UTF-8 (a BOM is tolerated); any other bytes are ``manual_invalid``.
* ``.docx`` — a zip, read with the stdlib. The archive is bounded BEFORE extraction (entry count and the
  sum of the entries' declared expanded sizes), so a zip bomb is refused rather than expanded, and the
  document text is taken from ``word/document.xml`` with the stdlib XML parser.
* ``.pdf`` — text PDFs only (``pypdf``): an encrypted PDF is refused, and a PDF that yields no text at all
  (a scan) is ``manual_empty`` rather than an empty manual.

Anything else, including a ``.docx`` that is really an encrypted/legacy OLE file, is ``manual_unsupported``.
"""

from __future__ import annotations

import base64
import binascii
import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Annotated

from pydantic import Field, ValidationError

from backend.app.api_contracts import UUIDString
from backend.app.errors import ApiFailure
from backend.app.guide_contracts import MATERIAL_TEXT_MAX, MATERIAL_TITLE_MAX, MaterialInput
from backend.app.visual_contracts import WireModel

#: The decoded source cap (5 MiB). A larger file is refused whole, never truncated.
MANUAL_SOURCE_MAX_BYTES = 5 * 1024 * 1024
#: The encoded request cap (~7 MiB): base64 inflates by 4/3 plus the JSON envelope and a data-URL prefix.
MANUAL_REQUEST_MAX_BYTES = 7 * 1024 * 1024
#: A ``.docx`` is a zip: bounded so an archive bomb cannot be expanded here.
MANUAL_DOCX_MAX_ENTRIES = 2048
MANUAL_DOCX_MAX_EXPANDED_BYTES = 32 * 1024 * 1024
#: Plain-text suffixes read as strict UTF-8.
TEXT_SUFFIXES = (".txt", ".md")
#: WordprocessingML namespace and the OLE/CFB magic an encrypted or legacy Office file starts with.
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

ManualBase64 = Annotated[str, Field(min_length=1, max_length=MANUAL_REQUEST_MAX_BYTES)]
ManualFileName = Annotated[str, Field(min_length=1, max_length=200)]


class ManualImportRequest(WireModel):
    """One local import: the current app session and the raw file bytes, base64-encoded.

    ``content_base64`` may be a data URL (``data:...;base64,``), which the client's ``FileReader`` produces.
    """

    session_id: UUIDString
    filename: ManualFileName
    content_base64: ManualBase64


def _display_name(filename: str) -> str:
    """The file's own name (no directory) trimmed to ``MATERIAL_TITLE_MAX``; ``manual`` if it is all separators."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return (base or "manual")[:MATERIAL_TITLE_MAX]


def _suffix(filename: str) -> str:
    """The supported suffix of ``filename`` (lowercased, path stripped), or ``""`` for anything else."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    for suffix in (".txt", ".md", ".pdf", ".docx"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return suffix
    return ""


def _decode(encoded: str) -> bytes:
    """The decoded bytes, or ``manual_invalid``/``manual_too_large``/``manual_empty``."""
    clean = encoded.split(",", 1)[1].strip() if encoded.startswith("data:") and "," in encoded else encoded.strip()
    try:
        raw = base64.b64decode(clean, validate=True)
    except (binascii.Error, ValueError):
        raise ApiFailure(422, "manual_invalid") from None
    if len(raw) > MANUAL_SOURCE_MAX_BYTES:
        raise ApiFailure(413, "manual_too_large")
    if not raw:
        raise ApiFailure(422, "manual_empty")
    return raw


def _text_suffix(raw: bytes) -> str:
    """Strict UTF-8 text (a BOM tolerated): any other byte sequence is ``manual_invalid``."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ApiFailure(422, "manual_invalid") from None


class _ManualTreeBuilder(ET.TreeBuilder):
    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        # Parser-level rejection also covers UTF-16 declarations, unlike a raw byte search.
        raise ApiFailure(422, "manual_invalid")


def _docx_text(document: bytes) -> str:
    """The visible text of ``word/document.xml``: one blank-line-separated block per paragraph.

    A DTD or an entity declaration is refused: the documents this lane accepts never need one, and the XML
    parser would otherwise expand internal entities (an XXE/billion-laughs surface) during extraction.
    """
    try:
        root = ET.fromstring(document, parser=ET.XMLParser(target=_ManualTreeBuilder()))
    except ET.ParseError:
        raise ApiFailure(422, "manual_invalid") from None
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_W_NS}p"):
        line = "".join(node.text or "" for node in paragraph.iter(f"{_W_NS}t")).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _docx(raw: bytes) -> str:
    """The text of a ``.docx``, with the archive bounded before anything is expanded."""
    if raw.startswith(_CFB_MAGIC):
        # A password-protected or legacy Word file renamed to .docx: unsupported, not a broken zip.
        raise ApiFailure(422, "manual_unsupported")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise ApiFailure(422, "manual_invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > MANUAL_DOCX_MAX_ENTRIES:
                raise ApiFailure(422, "manual_invalid")
            if sum(entry.file_size for entry in entries) > MANUAL_DOCX_MAX_EXPANDED_BYTES:
                raise ApiFailure(413, "manual_too_large")
            try:
                document = archive.read("word/document.xml")
            except KeyError:
                raise ApiFailure(422, "manual_invalid") from None
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, ValueError):
        raise ApiFailure(422, "manual_invalid") from None
    return _docx_text(document)


def _pdf(raw: bytes) -> str:
    """The extracted text of a text PDF; an encrypted PDF or a scan is refused, never returned empty."""
    try:
        from pypdf import PdfReader
    except ImportError:  # a deployment that did not install the optional text-PDF reader
        raise ApiFailure(503, "service_unavailable") from None
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ApiFailure(422, "manual_unsupported")
        if len(reader.pages) > 100:
            raise ApiFailure(413, "manual_too_large")
        pages: list[str] = []
        total = 0
        expanded = 0
        for page in reader.pages:
            contents = page.get_contents()
            expanded += len(contents.get_data()) if contents is not None else 0
            if expanded > MANUAL_DOCX_MAX_EXPANDED_BYTES:
                raise ApiFailure(413, "manual_too_large")
            text = page.extract_text() or ""
            if not text.strip():
                # Do not silently drop a scanned page from an otherwise text-based manual.
                raise ApiFailure(422, "manual_empty")
            total += len(text)
            pages.append(text)
            if total > MATERIAL_TEXT_MAX:
                raise ApiFailure(422, "manual_too_long")
    except ApiFailure:
        raise
    except Exception:  # pypdf raises a variety of parse errors for a malformed file
        raise ApiFailure(422, "manual_invalid") from None
    return "\n\n".join(pages)


def _finish(filename: str, text: str) -> MaterialInput:
    """The bounded ``MaterialInput``, or ``manual_empty``/``manual_too_long``/``manual_invalid``."""
    text = text.strip()
    if not text:
        raise ApiFailure(422, "manual_empty")
    if len(text) > MATERIAL_TEXT_MAX:
        raise ApiFailure(422, "manual_too_long")
    try:
        return MaterialInput(title=_display_name(filename), text=text)
    except ValidationError:
        raise ApiFailure(422, "manual_invalid") from None


def extract_manual(filename: str, content_base64: str) -> MaterialInput:
    """Extract one uploaded file into a ``MaterialInput``. CPU-only, offline, bounded; fails closed."""
    suffix = _suffix(filename)
    if not suffix:
        raise ApiFailure(422, "manual_unsupported")
    raw = _decode(content_base64)
    if suffix in TEXT_SUFFIXES:
        text = _text_suffix(raw)
    elif suffix == ".docx":
        text = _docx(raw)
    else:  # ".pdf"
        text = _pdf(raw)
    return _finish(filename, text)
