"""Attachment-link parser of the old municipal transparency site (`core.sources.comune_archive`), on a synthetic page shaped like the real one."""
from cifresignificative.core.sources import comune_archive as ca

BASE = "https://x.example/flex/cm/pages/ServeAttachment.php/L/IT/D/1%252F3%252F9%252FD.af94/P/BLOB%3AID%3D49680/E/"
PAGE = f"""<ul>
<li><a href="{BASE}csv?mode=download" title="x">Entrate del rendiconto 2021  (3.81 KB)</a></li>
<li><a href='{BASE}pdf?mode=download'><span>Rendiconto 2021 (in base al D.Lgs. 118/2011)</span> (10.3 MB)</a></li>
<li><a href="{BASE}csv?mode=download">Entrate del rendiconto 2021  (3.81 KB)</a></li>
<li><a href="https://x.example/trasparenza/bilancioconsuntivo2020">2020</a></li></ul>"""


def test_parse_attachments_keeps_downloads_once_with_clean_titles():
    out = ca.parse_attachments(PAGE)
    assert [(e, t) for _, e, t in out] == [("csv", "Entrate del rendiconto 2021"), ("pdf", "Rendiconto 2021 (in base al D.Lgs. 118/2011)")]
    assert out[1][0].endswith("pdf?mode=download") and out[0][0].endswith("csv?mode=download")


def test_page_name_uses_last_segment_or_page_id():
    assert ca.page_name("/trasparenza/bilancioconsuntivo2021") == "bilancioconsuntivo2021"
    assert ca.page_name("/flex/cm/pages/ServeBLOB.php/L/IT/IDPagina/13304") == "IDPagina13304"
