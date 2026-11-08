"""Sub-national geography for Italian municipal and provincial deep-dives (Emilia-Romagna & Cesena).

Provides the canonical registry of the 13 Emilia-Romagna benchmark municipalities (Cesena '040007', the 6 core peers,
and the 6 extended regional peers), their NUTS3 (2021) and NUTS2 mappings, and helper functions to standardise ISTAT
6-digit municipality codes across sources (e.g. OpenCoesione's split COD_PROVINCIA + COD_COMUNE vs MEF/ISTAT 6-digit strings).
"""
from dataclasses import dataclass

import pandas as pd

CESENA = "040007"
FORLI_CESENA_NUTS3 = "ITH58"
EMILIA_ROMAGNA_NUTS2 = "ITH5"


@dataclass(frozen=True)
class Comune:
    id: str          # 6-digit ISTAT code (e.g. '040007')
    name: str
    nuts3: str       # NUTS 2021 5-char code (e.g. 'ITH58')
    prov_code: str   # 3-digit ISTAT province code (e.g. '040')
    prov_sigla: str  # 2-letter province code (e.g. 'FC')
    catastale: str   # 4-char cadastral/Belfiore code (e.g. 'C573')
    core_peer: bool  # True for the 6 primary peers in the Cesena benchmark design


ER_COMUNI: tuple[Comune, ...] = (
    Comune("040007", "Cesena", "ITH58", "040", "FC", "C573", False),
    Comune("040012", "Forlì", "ITH58", "040", "FC", "D704", True),
    Comune("099014", "Rimini", "ITH59", "099", "RN", "H294", True),
    Comune("039014", "Ravenna", "ITH57", "039", "RA", "H199", True),
    Comune("037032", "Imola", "ITH55", "037", "BO", "E289", True),
    Comune("036005", "Carpi", "ITH54", "036", "MO", "B819", True),
    Comune("035033", "Reggio nell'Emilia", "ITH53", "035", "RE", "H223", True),
    Comune("038008", "Ferrara", "ITH56", "038", "FE", "D548", False),
    Comune("036023", "Modena", "ITH54", "036", "MO", "F257", False),
    Comune("034027", "Parma", "ITH52", "034", "PR", "G337", False),
    Comune("033032", "Piacenza", "ITH51", "033", "PC", "G535", False),
    Comune("039010", "Faenza", "ITH57", "039", "RA", "D458", False),
    Comune("037006", "Bologna", "ITH55", "037", "BO", "A944", False),
)

COMUNI_BY_ID: dict[str, Comune] = {c.id: c for c in ER_COMUNI}
COMUNI_BY_CATASTALE: dict[str, Comune] = {c.catastale: c for c in ER_COMUNI}
ALL_COMUNI_IDS: tuple[str, ...] = tuple(c.id for c in ER_COMUNI)
CORE_PEER_IDS: tuple[str, ...] = tuple(c.id for c in ER_COMUNI if c.core_peer)
EXTENDED_PEER_IDS: tuple[str, ...] = tuple(c.id for c in ER_COMUNI if c.id != CESENA)


@dataclass(frozen=True)
class Province:
    name: str
    nuts3: str
    prov_sigla: str
    prov_code: str
    pop_2024: int


ER_PROVINCES: dict[str, Province] = {
    "Piacenza": Province("Piacenza", "ITH51", "PC", "033", 286211),
    "Parma": Province("Parma", "ITH52", "PR", "034", 455471),
    "Reggio Emilia": Province("Reggio Emilia", "ITH53", "RE", "035", 528735),
    "Modena": Province("Modena", "ITH54", "MO", "036", 709325),
    "Bologna": Province("Bologna", "ITH55", "BO", "037", 1017806),
    "Ferrara": Province("Ferrara", "ITH56", "FE", "038", 339598),
    "Ravenna": Province("Ravenna", "ITH57", "RA", "039", 386527),
    "Forlì-Cesena": Province("Forlì-Cesena", "ITH58", "FC", "040", 392656),
    "Rimini": Province("Rimini", "ITH59", "RN", "099", 340629),
}
ER_PROVINCES_BY_NUTS3: dict[str, Province] = {p.nuts3: p for p in ER_PROVINCES.values()}
ER_PROVINCES_BY_SIGLA: dict[str, Province] = {p.prov_sigla: p for p in ER_PROVINCES.values()}


@dataclass(frozen=True)
class BeltComune:
    id: str          # 6-digit ISTAT code (e.g. '040008')
    name: str
    catastale: str   # 4-char cadastral/Belfiore code (e.g. 'C574')
    belt_group: str  # 'unione_valle_savio', 'rubicone_mare', 'via_emilia_west'


BELT_GROUP_LABELS: dict[str, str] = {
    "cesena_core": "Cesena (core municipality)",
    "unione_valle_savio": "Unione Valle del Savio (mountain/hill belt)",
    "rubicone_mare": "Unione Rubicone e Mare (coastal/industrial belt)",
    "via_emilia_west": "Via Emilia West (Forlimpopoli & Bertinoro)",
}

# The 16 municipalities forming Cesena's surrounding functional urban area / Comprensorio Cesenate in Forlì-Cesena (ITH58)
CESENA_BELT_COMUNI: tuple[BeltComune, ...] = (
    # Unione dei Comuni Valle del Savio (the other 5 municipalities besides Cesena)
    BeltComune("040001", "Bagno di Romagna", "A565", "unione_valle_savio"),
    BeltComune("040020", "Mercato Saraceno", "F139", "unione_valle_savio"),
    BeltComune("040028", "Montiano", "F668", "unione_valle_savio"),
    BeltComune("040044", "Sarsina", "I444", "unione_valle_savio"),
    BeltComune("040050", "Verghereto", "L764", "unione_valle_savio"),
    # Unione Rubicone e Mare (9 municipalities of the eastern Cesena commuter & industrial belt)
    BeltComune("040008", "Cesenatico", "C574", "rubicone_mare"),
    BeltComune("040015", "Gambettola", "D899", "rubicone_mare"),
    BeltComune("040016", "Gatteo", "D935", "rubicone_mare"),
    BeltComune("040018", "Longiano", "E675", "rubicone_mare"),
    BeltComune("040041", "San Mauro Pascoli", "I027", "rubicone_mare"),
    BeltComune("040045", "Savignano sul Rubicone", "I472", "rubicone_mare"),
    BeltComune("040004", "Borghi", "B001", "rubicone_mare"),
    BeltComune("040037", "Roncofreddo", "H542", "rubicone_mare"),
    BeltComune("040046", "Sogliano al Rubicone", "I779", "rubicone_mare"),
    # Immediate western Via Emilia corridor between Cesena and Forlì
    BeltComune("040003", "Bertinoro", "A809", "via_emilia_west"),
    BeltComune("040013", "Forlimpopoli", "D705", "via_emilia_west"),
)

BELT_BY_ID: dict[str, BeltComune] = {c.id: c for c in CESENA_BELT_COMUNI}
BELT_BY_CATASTALE: dict[str, BeltComune] = {c.catastale: c for c in CESENA_BELT_COMUNI}
BELT_COMUNI_IDS: tuple[str, ...] = tuple(c.id for c in CESENA_BELT_COMUNI)
COMPRENSORIO_COMUNI_IDS: tuple[str, ...] = (CESENA, *BELT_COMUNI_IDS)


# NUTS 2010 -> NUTS 2021 translation for Emilia-Romagna provinces (ISTAT still uses ITD5* in some legacy SDMX flows)
NUTS3_2010_TO_2021: dict[str, str] = {f"ITD5{k}": f"ITH5{k}" for k in range(1, 10)}


def format_comune_id(code: str | int) -> str:
    """Zero-pad an integer or string ISTAT municipality code to 6 digits ('40007' -> '040007')."""
    s = str(code).strip()
    s = s.removesuffix(".0")
    return s.zfill(6)


def from_prov_comune(prov: pd.Series, comune: pd.Series) -> pd.Series:
    """Build 6-digit ISTAT comune_id from separate province ('040' or '40') and municipality ('007' or '7') columns (as in OpenCoesione)."""
    p = prov.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
    c = comune.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(3)
    valid = p.str.fullmatch(r"\d{3}") & c.str.fullmatch(r"\d{3}") & (p != "000") & (c != "000")
    return (p + c).where(valid, None)


def harmonise_nuts3(codes: pd.Series) -> pd.Series:
    """Translate legacy NUTS 2010 province codes (ITD58) to NUTS 2021 (ITH58)."""
    return codes.map(lambda c: NUTS3_2010_TO_2021.get(str(c), str(c)))
