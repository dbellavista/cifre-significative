"""Unit tests for the municipal transparency-portal scraper (`core.sources.trasparenza`), on small synthetic pages shaped like the real ones."""
import base64
from pathlib import Path

import pytest
import requests

from cifresignificative.core.sources import trasparenza as t

URL = "https://x.example/web/trasparenza/papca-g?p_p_id=p&p_p_lifecycle=2&p_p_resource_id=downloadAllegato&_p_id=%s"


def enc(url: str) -> str:
    return base64.b64encode(url.encode()).decode()


MENU = '''<li data-resource="Bilanci" data-parentid="1" data-mainurl=""></li>
<li data-resource="Bilanci - Bilancio preventivo e consuntivo" data-parentid="2" data-mainurl="/web/trasparenza/papca-g/-/papca/igrid/41683"></li>
<li data-resource="Personale - OIV" data-parentid="3" data-mainurl="/web/trasparenza/pas/-/pas/igrid/41708"></li>'''

LIST = '''<form id="form_page_size" action="https://x.example/web/trasparenza/papca-g?p_auth=T&amp;a=eseguiPaginazione" method="post"></form>
<li class="active"><span>Pagina 1 di 6 (101 risultati)</span></li>
<table class="master-detail-list-table table"><tr><td>Tipo Atto</td><td>Anno e Numero Atto</td><td>Oggetto</td><td>Periodo</td><td></td></tr>
<tr><td>ATTI / DELIBERE DI CONSIGLIO</td><td>2023/45</td><td>RENDICONTO ANNO 2022. APPROVAZIONE</td><td>10/05/2023 31/12/2028</td><td>7</td>
<td><a href="https://x.example/web/trasparenza/papca-g/-/papca/display/111?p_auth=T&amp;p_p_state=pop_up">v</a></td></tr>
<tr><td>AMMINISTRAZIONE TRASPARENTE / PUBBLICAZIONE</td><td>0/0</td><td>DOCUMENTO UNICO DI PROGRAMMAZIONE 2027</td><td>06/10/2026 31/12/2031</td><td>1</td>
<td><a href="https://x.example/web/trasparenza/papca-g/-/papca/display/222?p_auth=T">v</a></td></tr></table>'''

DETAIL = f'''Categoria ATTI AMMINISTRATIVI Sottocategoria DELIBERE DI CONSIGLIO Anno atto 2023 Numero atto 45 Oggetto RENDICONTO ANNO 2022. APPROVAZIONE Data atto 01/05/2023 Numero allegati 2
<table class="table table-striped table-hover allegati-table">
<tr data-chiave-allegato="900" data-mimetype="application/pdf"><td>delibera.pdf</td><td>Documento principale</td><td><a onclick="window.open(atob('{enc(URL % 1001)}'), '_blank')" href="javascript:void(0)"></a></td></tr>
<tr data-chiave-allegato="901" data-mimetype="application/pdf"><td>allegato a</td><td>Allegato parte integrante</td><td><a onclick="window.open(atob('{enc(URL % 1002)}'), '_blank')" href="javascript:void(0)"></a></td></tr></table>'''

SOGGETTI = '''<table class="table-albi-soggetto table"><tr><th class="table-albi-soggetto-header"><div>Relazione del revisore anno 2024</div></th></tr>
<tr><td>Data di pubblicazione 11/06/2026 <a href="https://x.example/web/trasparenza/pas?p_p_resource_id=downloadAllegato&amp;_p_id=555&amp;x=1">dl</a></td></tr></table>
<table class="table-albi-soggetto table"><tr><th class="table-albi-soggetto-header"><div>Altro</div></th></tr><tr><td>nessun file</td></tr></table>'''


def test_decode_download_url_round_trip():
    assert t.decode_download_url(enc(URL % 7)) == URL % 7


def test_parse_menu_keeps_only_leaves_with_a_url():
    cat = t.parse_menu(MENU)
    assert set(cat) == {"Bilanci - Bilancio preventivo e consuntivo", "Personale - OIV"}
    assert cat["Personale - OIV"].endswith("/pas/-/pas/igrid/41708")
    assert t.select_leaves(cat, ("Bilanci",)).keys() == {"Bilanci - Bilancio preventivo e consuntivo"}


@pytest.mark.parametrize(("number", "published", "year"), [("2023/45", "x", 2023), ("0/0", "06/10/2026 31/12/2031", 2026), ("0000/1", "01/02/2021 -", 2021), ("", "", None)])
def test_year_of(number, published, year):
    assert t.year_of(number, published) == year


def test_parse_list_rows_position_and_form():
    rows, pos, form = t.parse_list(LIST, "Bilanci")
    assert pos == (1, 6, 101) and form is not None and form.endswith("a=eseguiPaginazione")
    assert [r.doc_id for r in rows] == ["111", "222"]
    assert rows[0].subject == "RENDICONTO ANNO 2022. APPROVAZIONE" and rows[0].n_attachments == 7 and rows[0].year == 2023
    assert rows[1].year == 2026  # act "0/0": the year comes from the publication date


def test_parse_detail_metadata_and_decoded_attachments():
    d = t.parse_detail(DETAIL, "111", "Bilanci")
    assert d.meta["Oggetto"] == "RENDICONTO ANNO 2022. APPROVAZIONE" and d.meta["Anno atto"] == "2023" and d.meta["Numero allegati"] == "2"
    assert [a.att_id for a in d.attachments] == ["900", "901"]
    assert d.attachments[0].url == URL % 1001 and d.attachments[1].description == "Allegato parte integrante"


def test_parse_soggetti_blocks_with_and_without_files():
    items = t.parse_soggetti(SOGGETTI, "Controlli")
    assert [i.meta["Oggetto"] for i in items] == ["Relazione del revisore anno 2024", "Altro"]
    assert items[0].meta["Data documento"] == "11/06/2026" and items[0].attachments[0].att_id == "555" and items[1].attachments == []


def test_slug_is_ascii_and_bounded():
    assert t.slug("Relazione è già così: 2024/1.pdf") == "Relazione_e_gia_cosi_2024_1.pdf"
    assert len(t.slug("a" * 300)) == 80 and t.slug("///") == "file"


def test_filename_from_content_disposition_and_mimetype():
    r = requests.Response()
    r.headers["Content-Disposition"] = "attachment; filename*=UTF-8''Rendiconto%202023.pdf"
    assert t.filename_from(r, t.Attachment("9", "x", "", "", "u")) == "9_Rendiconto_2023.pdf"
    r2 = requests.Response()
    r2.headers["Content-Type"] = "application/pdf"
    assert t.filename_from(r2, t.Attachment("10", "allegato b", "", "", "u")) == "10_allegato_b.pdf"


class FakeResponse:
    def __init__(self, status: int = 200, content: bytes = b"%PDF-1.4 x", headers: dict | None = None):
        self.status_code, self.content, self.headers, self.text = status, content, headers or {}, content.decode("latin1")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_site_retries_a_dropped_connection_then_succeeds(tmp_path: Path, monkeypatch):
    calls = []

    def flaky(self, method, url, **kw):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ConnectionError("dropped")
        return FakeResponse()

    monkeypatch.setattr(requests.Session, "request", flaky)
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    site = t.Site("https://x.example", raw_dir=tmp_path)
    assert site.get("/a").status_code == 200 and len(calls) == 3


def test_site_gives_up_after_the_retries(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: FakeResponse(503))
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="giving up"):
        t.Site("https://x.example", raw_dir=tmp_path, retries=2).get("/a")


def test_download_writes_once_and_records_hash(tmp_path: Path, monkeypatch):
    n = []
    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: n.append(u) or FakeResponse())
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    site = t.Site("https://x.example", raw_dir=tmp_path)
    doc, att = t.Detail("111", "Bilanci", {"Oggetto": "RENDICONTO"}), t.Attachment("900", "delibera.pdf", "", "application/pdf", "https://x.example/f")
    row = t.download(site, doc, att)
    again = t.download(site, doc, att)
    assert len(n) == 1 and row == again and row["pdf"] is True and row["bytes"] == 10
    assert (site.dir / str(row["file"])).exists() and len(str(row["sha256"])) == 64
    p = t.write_manifest(site, [row])
    assert "900" in p.read_text()


def test_signed_envelopes_are_skipped_unless_asked():
    pdf, p7m = t.Attachment("1", "a", "", "application/pdf", "u"), t.Attachment("2", "a.pdf.p7m", "", "application/x-pkcs7-mime", "u")
    assert t.wanted(pdf) and not t.wanted(p7m) and t.wanted(p7m, with_signed=True)
