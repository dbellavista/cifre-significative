# Cifre significative: codice e dati

Codice, notebook e tabelle dietro la serie di post [Cifre significative](https://danielebellavista.substack.com/) (Italia confrontata con gli altri paesi dell'Unione Europea, su dati pubblici e con metodi verificabili). _English: reproducible pipeline and analysis comparing Italy with the other EU27 members._

Dal 2026 c'è anche la sezione [Cifre Significative - Cesena edition](https://danielebellavista.substack.com/s/cifre-significative-cesena/), con lo studio su Cesena (`studies/cesena/`).

Tutto è **descrittivo, mai causale**: le associazioni non dimostrano cause. Le sei regole editoriali della serie sono nella pagina "About" del Substack.

## Gli articoli
Un articolo per risultato; i numeri citati si ricalcolano con il codice di questo repository.

| serie | articolo | dove sono i notebook |
|---|---|---|
| Cifre significative (Italia) | [L'Italia non è povera, è ferma](https://danielebellavista.substack.com/p/litalia-non-e-povera-e-ferma) | `studies/italy_productivity/notebooks/` |
| Significant Figures (Italy) | [Italy is not poor. It has stopped growing.](https://danielebellavista.substack.com/p/italy-is-not-poor-it-has-stopped) | `studies/italy_productivity/notebooks/` |
| Cifre significative (Italia) | [Un problema di produttività dentro ogni settore](https://danielebellavista.substack.com/p/un-problema-di-produttivita-dentro) | `studies/italy_productivity/notebooks/` |
| Cifre significative (Italia) | [Le piccole imprese pesano, ma non spiegano la frenata](https://danielebellavista.substack.com/p/le-piccole-imprese-pesano-ma-non) | `studies/italy_productivity/notebooks/` |
| Cifre significative (Italia) | [La regolamentazione dei mercati: nessuna conferma dai dati](https://danielebellavista.substack.com/p/la-regolamentazione-dei-mercati-nessuna) | `studies/italy_productivity/notebooks/35_regulation_productivity.ipynb` |
| Cesena edition | [Cesena cresce? Dipende da cosa misuriamo](https://danielebellavista.substack.com/p/cesena-cresce-dipende-da-cosa-misuriamo) | `studies/cesena/notebooks/` |

## Cosa c'è
| cartella | contenuto |
|---|---|
| `src/cifresignificative/` | pipeline (scarico, pulizia), analisi e cruscotto Streamlit |
| `studies/italy_productivity/notebooks/` | 42 notebook eseguiti, con i risultati e le note a margine |
| `studies/italy_productivity/reports/findings.md` | sintesi dei risultati con numeri e limiti; i grafici sono in `studies/italy_productivity/reports/figures/` |
| `studies/cesena/` | studio su Cesena: 5 notebook eseguiti, `reports/findings.md` (risultati con numeri e limiti), README con fonti e insidie, `reports/income_evidence/` (tabelle dei redditi usate nel post di lancio) |
| `docs/` | metodologia, schema dei dati, catalogo degli indicatori, quadro delle analisi |
| `data/external/*.csv` | tabelle curate a mano (governi, politiche, glossario, tabelle compilate da fonti aperte) |
| `tests/` | test della pipeline e dei metodi statistici |

## Riprodurre
```
uv sync
uv run python -m cifresignificative.core.fetch           # scarica i dati dalle fonti originali (Eurostat, AMECO, OCSE, ...)
uv run python -m cifresignificative.core.clean.build     # costruisce data/processed/*.parquet
uv run inv test.run
```
Alcune fonti non si scaricano da sole o non sono ridistribuibili: [`docs/reproducibility.md`](docs/reproducibility.md) dice quali notebook ne dipendono e come procurarsele.

## Licenze
- **Codice:** MIT (`LICENSE`).
- **Grafici, tabelle in `data/external/*.csv` e testi** (`reports/`, `docs/`): CC BY 4.0, citando l'autore (`LICENSE-CC-BY-4.0.md`). I dati originali di terze parti restano soggetti alle licenze delle fonti.

## Dati
- **I dati grezzi non sono ridistribuiti**: gli script li scaricano dalle fonti originali, che restano proprietà dei rispettivi autori (Eurostat, Istat, Commissione europea, OCSE, ENEA, CNEL, MEF, ...). Citare sempre la fonte originale.
- Alcuni file di terze parti (documenti Istat e Camera, serie giornaliere dei rendimenti, tabelle CNEL) servono a certi notebook ma non sono inclusi; `data/external/source/*/README.md` spiega da dove vengono. I notebook che li usano si possono riprodurre solo scaricandoli di nuovo.
- Le tabelle in `data/external/*.csv` sono lavoro dell'autore o compilazioni da fonti pubbliche, con l'indicazione della fonte nelle colonne `evidence`/`source` quando esiste.
- **Termini di riuso delle tabelle compilate da terzi** (in `data/external/*.csv`):
  - `ec_forecasts.csv`: previsioni della Commissione europea (CC BY 4.0, decisione 2011/833/UE; dati estratti dagli annex delle previsioni d'autunno, modifiche indicate).
  - `piaac.csv`, `pisa_2022.csv`: dati OCSE (Survey of Adult Skills 2023 via OECD Education GPS; PISA 2022 Results Vol. I). Riuso consentito anche a fini commerciali con attribuzione (OECD (anno), nome del dataset, fonte, URL, data di accesso), da riportare in ogni riuso; nessun avallo dell'OCSE.
  - `cnel_workers.csv` (conteggi CNEL-INPS di datori e dipendenti per contratto) **non è incluso**: le note legali del CNEL ammettono il riuso solo non commerciale e dei contenuti integri. Si rigenera con `cnel_workers build` dai file Excel scaricati dall'archivio CNEL (`data/external/source/cnel/README.md`). Le date di rinnovo (`cnel_contracts.csv`) vengono invece dall'API open data del CNEL (IODL 2.0).
  - `conto_annuale_headcount.csv`: Ragioneria generale dello Stato, Conto annuale; fonte da citare.
  - `rating_actions.csv`: solo fatti (data, agenzia, voto e outlook) tratti dalle pagine pubbliche di S&P, Moody's, Fitch e DBRS; nessun testo o analisi delle agenzie, che restano titolari dei propri contenuti e dei marchi.
- Se trovi un errore, apri una issue o scrivi all'autore: il numero verrà corretto e il post aggiornato.
