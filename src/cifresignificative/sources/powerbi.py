"""Decode Power BI 'querydata' responses (DSR format) captured from public embedded reports into DataFrames."""
import pandas as pd


def decode(response: dict) -> pd.DataFrame:
    """One querydata JSON response -> DataFrame (columns named after the report's fields).

    Row encoding: `C` holds only the values that are neither repeated from the previous row (bit set in `R`) nor null
    (bit set in `Ø`); dictionary-coded columns store an index into `ValueDicts`. The schema `S` appears on the first row only.
    """
    frames = []
    for result in response.get("results", []):
        data = result.get("result", {}).get("data", {})
        names_by_id = {s["Value"]: s.get("Name", s["Value"]) for s in data.get("descriptor", {}).get("Select", [])}
        for ds in data.get("dsr", {}).get("DS", []):
            dicts = ds.get("ValueDicts", {})
            for ph in ds.get("PH", []):
                for rows in ph.values():
                    schema, prev, out = None, None, []
                    for row in rows:
                        schema = row.get("S", schema)
                        if schema is None:
                            continue
                        n = len(schema)
                        c, r, nul = iter(row.get("C", [])), row.get("R", 0), row.get("Ø", 0)
                        vals = []
                        for i in range(n):
                            if r & (1 << i) and prev is not None:
                                v = prev[i]
                            elif nul & (1 << i):
                                v = None
                            else:
                                v = next(c, None)
                                dn = schema[i].get("DN")
                                if dn is not None and v is not None:
                                    v = dicts[dn][v]
                            vals.append(v)
                        prev = vals
                        out.append(vals)
                    if out and schema is not None:
                        cols = [names_by_id.get(sc["N"], sc["N"]) for sc in schema]
                        frames.append(pd.DataFrame(out, columns=cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decode_matrix(response: dict) -> pd.DataFrame:
    """Country x year responses (primary group + secondary group in `SH`): long DataFrame [group, secondary, measure columns].

    Each `X` item is one secondary-group member (in order, or at position `I` when earlier ones are empty). Repeat/null
    bit-masks (`R`, `Ø`) refer to the previously decoded item, across groups.
    """
    frames = []
    for result in response.get("results", []):
        data = result.get("result", {}).get("data", {})
        select = {s["Value"]: s.get("Name", s["Value"]) for s in data.get("descriptor", {}).get("Select", [])}
        for ds in data.get("dsr", {}).get("DS", []):
            secondary = [m for sh in ds.get("SH", []) for part in sh.values() for m in part]
            sec_key = next((k for m in secondary for k in m if k != "S"), None)
            sec_values = [m.get(sec_key) for m in secondary]
            rows, prev, schema = [], None, None
            for ph in ds.get("PH", []):
                for members in ph.values():
                    gschema = None
                    for member in members:
                        gschema = member.get("S", gschema)
                        gkey = next((k for k in member if k not in ("S", "X", "C", "R", "Ø")), None)
                        gval = member.get(gkey)
                        pos = 0
                        for item in member.get("X", []):
                            pos = item.get("I", pos)
                            schema = item.get("S", schema)
                            if schema is None:
                                continue
                            n, c, r, nul = len(schema), iter(item.get("C", [])), item.get("R", 0), item.get("Ø", 0)
                            vals = []
                            for i in range(n):
                                if r & (1 << i) and prev is not None:
                                    v = prev[i]
                                elif nul & (1 << i):
                                    v = None
                                else:
                                    v = next(c, None)
                                vals.append(v)
                            prev = vals
                            rows.append([gval, sec_values[pos] if pos < len(sec_values) else None, *vals])
                            pos += 1
            if rows and schema is not None:
                measures = [select.get(s["N"], s["N"]) for s in schema]
                gname = next((v for k, v in select.items() if k.startswith("G") and "Year" not in v), "group")
                frames.append(pd.DataFrame(rows, columns=[gname, "year", *measures]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
