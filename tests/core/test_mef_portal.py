"""MEF portal crawler: card parsing on a saved listing snippet and the download rules, without network access."""
import re
from pathlib import Path
from typing import cast

import requests

from cifresignificative.core.sources import mef_portal as P

CARD = """<div class="card card-primary card-primary--shadow mb-0 border-0"><div class="card-header "><h3 class=" weight-700 mb-0 py-2">
 2025 a.i. 2024 / IRPEF / Persone fisiche totali / Tutte le tipologie di contribuenti </h3></div><div class="card-body ">
 <div class="col"><h4 class=" weight-700 mb-0">Data:</h4><span class=" weight-400 fs-5 "> 23 Aprile 2026 &nbsp; </span></div>
 <div class="col"><h4 class=" weight-700 mb-0">Tematica:</h4><span class=" weight-400 fs-5 "> Combinazione multipla </span></div>
 <div class="col"><h4 class=" weight-700 mb-0">Classificazione:</h4><span class=" weight-400 fs-5 "> Ventili di reddito complessivo </span></div>
 <a href="index.php?tree=2025AAPFTOT0203&amp;export=3&amp;aggiornato=1">CSV</a></div></div>"""


def test_parse_cards_reads_title_fields_and_code():
    page = "<html>" + CARD + CARD.replace("2025AAPFTOT0203", "2024AETIPLD0104").replace("2025 a.i. 2024", "2024 a.i. 2023") + "<div class=\"card card-primary\"><h3>no link</h3></div></html>"
    cards = P.parse_cards(page)
    assert [c["code"] for c in cards] == ["2025AAPFTOT0203", "2024AETIPLD0104"]  # the card without an export link is skipped
    c = cards[0]
    assert c["year"] == 2025 and c["table"] == "AAPFTOT0203" and c["tematica"] == "Combinazione multipla" and c["classificazione"] == "Ventili di reddito complessivo" and c["title"].startswith("2025 a.i. 2024 / IRPEF")


def test_persons_filter_keeps_individuals_only():
    assert all(re.match(P.PERSONS, t) for t in ("AAPFTOT0203", "ABPFTIT0203", "AETIPLD0104")) and not re.match(P.PERSONS, "ACSPIMP01000")


def test_download_skips_existing_files_and_empty_bodies(tmp_path, monkeypatch):
    calls = []

    class S:
        def get(self, url, **kwargs):
            tree = kwargs["params"]["tree"]
            calls.append(tree)
            r = requests.Response()
            r.status_code = 200
            r._content = b"" if tree.startswith("2005") else b"a;b\n1;2\n"
            return r

    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    assert P.download("2005AAPFTOT0203", tmp_path, session=cast(requests.Session, S())) is None and not (tmp_path / "2005").exists()
    p = P.download("2024AAPFTOT0203", tmp_path, session=cast(requests.Session, S()))
    assert isinstance(p, Path) and p.read_bytes().startswith(b"a;b")
    assert P.download("2024AAPFTOT0203", tmp_path, session=cast(requests.Session, S())) == p and calls == ["2005AAPFTOT0203", "2024AAPFTOT0203"]  # the existing file is not requested again
