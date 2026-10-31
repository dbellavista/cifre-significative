# Riprodurre i risultati: quali dati servono e come ottenerli

I dati grezzi non sono nel repository: `fetch` li scarica dalle fonti originali in `data/raw/`, e `clean.build` costruisce le tabelle in `data/processed/`. Per alcune fonti non esiste un'API e il file va scaricato a mano (o non può essere ridistribuito): senza di esse la costruzione prosegue, salta quelle tabelle e stampa un avviso, e solo i notebook indicati sotto non si rieseguono.

```
uv sync
uv run python -m cifresignificative.core.fetch            # tutto ciò che si scarica da solo (Eurostat, AMECO, Banca Mondiale, FMI, OCSE, Istat, ...)
uv run python -m cifresignificative.core.clean.build      # costruisce data/processed/*.parquet; le fonti mancanti sono saltate con un avviso
uv run inv test.run
```

Questa guida non è stata verificata da un clone pulito fino all'ultimo notebook: se un passaggio non funziona, apri una issue.

## Fonti da procurarsi a parte

| notebook | cartella in `data/external/source/` | cosa serve | come ottenerlo |
|---|---|---|---|
| 27 (spread giornalieri e rating) | `daily_yields/` | rendimenti giornalieri a 10 anni di Italia, Germania, Spagna, Francia, Portogallo, Grecia (esportazioni di Investing.com, non ridistribuibili) | esportazione a mano dal sito, due file per paese (limite di 5.000 righe a esportazione); nome file `<paese>_10y_<periodo>.csv`; formato, date e controlli nel `README.md` della cartella |
| 27 | `eurosystem_holdings/` | acquisti del Sistema europeo di banche centrali per paese (PSPP e PEPP) | due CSV dalle pagine della BCE sui programmi di acquisto di attività; dettagli nel `README.md` della cartella |
| 09 (governance, evasione fiscale) | `evasion/` | "Relazione sull'economia non osservata e sull'evasione fiscale e contributiva", edizione 2025, del Dipartimento delle finanze (MEF) | PDF dal sito del MEF; il nome atteso è in `src/cifresignificative/core/sources/mef_gap.py` |
| 09 | `istat_unobserved/` | cartella di lavoro Istat "Tavole-Economia-non-osservata" (edizione 2025) | scaricarla dalla pagina del comunicato Istat sull'economia non osservata, sezione "Tavole"; vedi `docs/manual_data.md`, sezione 4 |
| 09, 32, 33 (dichiarazioni IRPEF) | `mef_irpef*/` (sei cartelle) | tabelle delle dichiarazioni del Dipartimento delle finanze | `uv run python scripts/fetch_mef_irpef.py list <classificazione>` e `get <classificazione>` per classi_reddito, ventili, regione, attivita, eta, fatturato (per `fatturato` solo `--tables '^AE'`); una richiesta al secondo |

Per questi notebook, senza la fonte, il resto del repository funziona: le celle già eseguite dei notebook mostrano i risultati.

## Fonti scaricate in automatico ma con un browser

`bready` (Banca Mondiale, B-READY Italia) e `procurement` (Single Market Scoreboard, appalti pubblici) si scaricano con `fetch` usando un browser automatico (Playwright). Se il sito chiede un passaggio manuale: `uv run python -m cifresignificative.core.fetch --interactive`. Il protocollo di scarico è in `docs/download_protocol.md`.

## Già incluse (nessun passaggio manuale)

- `data/external/*.csv`: tabelle curate a mano e compilazioni da fonti pubbliche (esclusa `cnel_workers.csv`, vedi `README.md`), tra cui `pisa_2022.csv`, `piaac.csv`, `rating_actions.csv` e `ec_forecasts.csv`. Come sono state fatte è in `docs/manual_data.md` e nei README delle cartelle in `data/external/source/`.
- I file sorgente da cui sono state ricavate non sono ridistribuiti; chi vuole rifare quel passaggio trova le istruzioni nei documenti citati.

## Cosa cambia se mancano le fonti

I numeri dei post che dipendono da queste fonti (spread giornalieri, evasione fiscale, dichiarazioni IRPEF) si possono controllare nei notebook già eseguiti, ma non rigenerare senza le fonti. Per tutto il resto, `fetch` e `build` bastano.
