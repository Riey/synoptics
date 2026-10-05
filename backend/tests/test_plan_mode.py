"""Plan-material invariants: the offline import (``POST /api/plan/manual``) and its hostile-input boundaries.

The import turns a user file into a bounded ``MaterialInput`` the client may send as a plan's ``materials``.
Nothing here calls a model, and a file that does not fit is refused whole — never truncated.
"""
import asyncio
import base64
import io
import zipfile

import pytest

from backend.tests.test_guide import (
    PLAN, api_client, current_plan, env, install, mutate, open_session, plan, prompt_text, review_step,
)

MANUAL = "컵을 화면 기준 오른쪽으로 옮긴다.\n컵을 책상 위에 내려놓는다."


def docx_bytes(xml, *, expanded_padding=0):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", xml)
        if expanded_padding:
            archive.writestr("word/media/bomb", b"0" * expanded_padding)
    return out.getvalue()


XML = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>컵을 감싼다.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>상자에 넣는다.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'


async def import_manual(api, headers, session_id, filename, data):
    return await api.post("/api/plan/manual", headers=headers, json={
        "session_id": session_id, "filename": filename,
        "content_base64": base64.b64encode(data).decode()})


def test_manual_import_returns_a_material_input_without_provider_call(env):
    """The ``.txt``/``.docx`` path extracts text and answers a ``MaterialInput`` (``title`` from the filename)."""
    async def exercise():
        stub = install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            for filename, data, title, expected in [
                ("절차.txt", MANUAL.encode(), "절차.txt", MANUAL),
                ("절차.docx", docx_bytes(XML), "절차.docx", "상자에 넣는다."),
            ]:
                result = await import_manual(api, headers, sid, filename, data)
                assert result.status_code == 200, result.text
                body = result.json()
                assert body["title"] == title and body["version"] is None
                assert expected in body["text"]
            assert stub.requests == []  # the import never reaches a provider

    asyncio.run(exercise())


def test_manual_import_never_truncates_and_obeys_its_bounds(env):
    """A 4000-character document is accepted whole; 4001 is refused (never cut). A long filename is a label,
    trimmed to the 80-character title bound."""
    async def exercise():
        install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            assert (await import_manual(api, headers, sid, "a.txt", ("가" * 4000).encode())).status_code == 200
            over = await import_manual(api, headers, sid, "a.txt", ("가" * 4001).encode())
            assert over.status_code == 422 and over.json() == {"error": "manual_too_long"}
            long_name = await import_manual(api, headers, sid, "가" * 120 + ".txt", MANUAL.encode())
            assert long_name.status_code == 200 and len(long_name.json()["title"]) == 80

    asyncio.run(exercise())


@pytest.mark.parametrize("filename,data", [
    ("empty.txt", b"  \n"),
    ("binary.txt", b"\xff\xfe\x00\x01"),
    ("archive.docx", b"not a zip"),
    ("entity.docx", docx_bytes('<!DOCTYPE x [<!ENTITY e "expanded">]>' + XML)),
    ("script.html", b"<script>alert(1)</script>"),
])
def test_manual_import_refuses_invalid_documents_without_provider_call(env, filename, data):
    async def exercise():
        stub = install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            result = await import_manual(api, headers, sid, filename, data)
            assert result.status_code in (413, 415, 422), result.text
            assert stub.requests == []

    asyncio.run(exercise())


def test_manual_import_obeys_origin_and_session_binding(env):
    async def exercise():
        install()
        async with api_client() as api:
            headers, sid = await open_session(api)
            body = {"session_id": sid, "filename": "a.txt", "content_base64": base64.b64encode(b"manual").decode()}
            wrong_origin = await api.post("/api/plan/manual", headers={**headers, "Origin": "https://other.invalid"}, json=body)
            assert wrong_origin.status_code == 403
            wrong_session = await api.post("/api/plan/manual", headers=headers,
                                           json={**body, "session_id": "00000000-0000-4000-8000-000000000000"})
            assert wrong_session.status_code == 401
    asyncio.run(exercise())


def test_an_imported_material_grounds_a_plan_end_to_end(env):
    """What the import returns is exactly what a plan start carries: the prompt renders it as the ``m1`` block
    and a step citing it keeps its evidence."""
    answer = mutate(PLAN, steps=[review_step(1, evidence={"material_id": "m1", "quote": "뚜껑을 연다"})])

    async def exercise():
        stub = install({"guide_plan": answer})
        async with api_client() as api:
            headers, sid = await open_session(api)
            imported = (await import_manual(api, headers, sid, "충전 매뉴얼.txt", "1. 뚜껑을 연다".encode())).json()
            assert imported["title"] == "충전 매뉴얼.txt"
            response = await plan(api, headers, sid, materials=[imported])
            assert response.status_code == 200, response.text
            assert response.json()["steps"][0]["evidence"]["quote"] == "뚜껑을 연다"
            assert '[m1] "충전 매뉴얼.txt"' in prompt_text(stub.requests[-1])
            # The material is stored with the plan (text only), never echoed by /plan/current.
            assert "materials" not in (await current_plan(api, headers, sid)).json()

    asyncio.run(exercise())


def test_utf16_document_cannot_bypass_entity_rejection():
    from backend.app.errors import ApiFailure
    from backend.app.plan_manual import extract_manual

    document = ('<?xml version="1.0" encoding="UTF-16"?>'
                '<!DOCTYPE x [<!ENTITY e "expanded">]>' + XML).encode("utf-16")
    with pytest.raises(ApiFailure) as error:
        extract_manual("entity.docx", base64.b64encode(docx_bytes(document)).decode())
    assert error.value.code == "manual_invalid"


def test_oversized_archive_is_rejected_before_text_extraction():
    from backend.app.errors import ApiFailure
    from backend.app.plan_manual import extract_manual

    archive = docx_bytes(XML, expanded_padding=33 * 1024 * 1024)
    with pytest.raises(ApiFailure) as error:
        extract_manual("expanded.docx", base64.b64encode(archive).decode())
    assert error.value.code == "manual_too_large"


def test_pdf_with_unreadable_page_is_not_silently_partial():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    from backend.app.errors import ApiFailure
    from backend.app.plan_manual import extract_manual

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Wrap the cup.) Tj ET")
    page[NameObject("/Contents")] = stream
    good = io.BytesIO()
    writer.write(good)
    extracted = extract_manual("text.pdf", base64.b64encode(good.getvalue()).decode())
    assert "Wrap the cup." in extracted.text and extracted.title == "text.pdf"
    writer.add_blank_page(width=300, height=300)
    mixed = io.BytesIO()
    writer.write(mixed)
    with pytest.raises(ApiFailure) as error:
        extract_manual("mixed.pdf", base64.b64encode(mixed.getvalue()).decode())
    assert error.value.code == "manual_empty"
