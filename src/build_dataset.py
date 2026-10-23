"""
Build a one-row-per-race modeling table from MEDSL U.S. House returns.

Input : raw MEDSL file (one row per candidate per party line per race)
Output: data/processed/house_races.csv (one row per state-district-year)

Usage:
    python src/build_dataset.py --input data/raw/<medsl_file>

Encoding convention: every signed feature is from the Democratic point of view
(+ favors D, - favors R). This keeps the model symmetric across parties.
"""

import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Party of the sitting president on Election Day (+1 = D, -1 = R).
PRES_PARTY = {
    1976: -1, 1978: 1, 1980: 1, 1982: -1, 1984: -1, 1986: -1, 1988: -1,
    1990: -1, 1992: -1, 1994: 1, 1996: 1, 1998: 1, 2000: 1, 2002: -1,
    2004: -1, 2006: -1, 2008: -1, 2010: 1, 2012: 1, 2014: 1, 2016: 1,
    2018: -1, 2020: -1, 2022: 1, 2024: 1,
}

# First elections held under post-census maps (lagged margins are unreliable).
# Mid-decade redraws in individual states are NOT captured here.
REDISTRICT_YEARS = {1982, 1992, 2002, 2012, 2022}

NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV"}

PSEUDO_CANDIDATE = r"BLANK|UNDER ?VOTE|OVER ?VOTE|EXHAUSTED|SCATTERING|^VOID"


# ---------------------------------------------------------------- loading
def repair_wrapped_lines(header: str, data: list[str]) -> list[str]:
    """
    Dataverse's regenerated .tab export wraps every data line in one pair of quotes
    and backslash-escapes the quotes and backslashes inside, e.g.
        "1976,ALABAMA,...,\\"WILLIAM L ...\\",REPUBLICAN,..."
    Undo that so each line is a normal CSV row again.
    """
    wrapped = sum(l.startswith('"') and l.endswith('"') for l in data)
    if header.startswith('"') or wrapped < 0.9 * len(data):
        return data                                            # file looks normal; leave it alone
    print(f"Repairing quote-wrapped export ({wrapped:,} of {len(data):,} lines)")
    return [re.sub(r"\\(.)", r"\1", l[1:-1]) if l.startswith('"') and l.endswith('"') else l
            for l in data]


def load_raw(path: Path) -> pd.DataFrame:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    lines = [l for l in text.splitlines() if l.strip()]
    header, data = lines[0], repair_wrapped_lines(lines[0], lines[1:])

    # Detect the delimiter from the header; file extensions on Dataverse aren't reliable.
    sep = "\t" if header.count("\t") > header.count(",") else ","
    print(f"Reading {path.name} with delimiter {'TAB' if sep == chr(9) else 'COMMA'}")
    df = pd.read_csv(io.StringIO("\n".join([header] + data)), sep=sep, low_memory=False)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    required = {"year", "state_po", "district", "candidate", "party", "candidatevotes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    return df


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().isin({"TRUE", "T", "1", "YES"})


def filter_general_elections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep, for each state-district-year, the rows of the election that decided the seat.
    Priority (highest wins):
        3  runoff                      (e.g. Texas 1996 December runoffs, Louisiana 2002)
        2  regular general election
        1  special general election    (e.g. Texas 2006 court-ordered redraws)
        0  special "primary"-stage     (Texas 1996 November open elections under a new map)
    Same-day special elections to fill a vacancy are dropped whenever a regular general exists.
    """
    df = df[df["state_po"] != "DC"]            # non-voting delegate, only present in some years
    # Ballot-counting artifacts recorded as "candidates" (blank/under/over votes, exhausted RCV ballots).
    pseudo = df["candidate"].astype(str).str.upper().str.contains(PSEUDO_CANDIDATE, regex=True)
    df = df[~pseudo]
    stage = df["stage"].astype(str).str.upper() if "stage" in df.columns else pd.Series("GEN", index=df.index)
    special = as_bool(df["special"]) if "special" in df.columns else pd.Series(False, index=df.index)
    runoff = as_bool(df["runoff"]) if "runoff" in df.columns else pd.Series(False, index=df.index)

    priority = np.select(
        [runoff, (stage == "GEN") & ~special, (stage == "GEN") & special, (stage == "PRI") & special],
        [3, 2, 1, 0], default=-1)
    df = df.assign(_priority=priority)
    df = df[df["_priority"] >= 0]              # regular (non-special) primaries are never used
    best = df.groupby(["year", "state_po", "district"])["_priority"].transform("max")
    df = df[df["_priority"] == best]
    df = df.assign(decided_by=df["_priority"].map({3: "runoff", 2: "general", 1: "special", 0: "special"}))
    df = df.drop(columns="_priority")
    # If a race reports both TOTAL and per-mode rows, keep only TOTAL to avoid double counting.
    if "mode" in df.columns:
        is_total = df["mode"].astype(str).str.upper() == "TOTAL"
        race_has_total = is_total.groupby([df["year"], df["state_po"], df["district"]]).transform("any")
        df = df[is_total | ~race_has_total]
    return df.copy()


# ---------------------------------------------------------------- cleaning
def major_party(p) -> str:
    p = str(p).upper().strip()
    if p.startswith("DEMOCRAT"):          # DEMOCRAT, DEMOCRATIC-FARMER-LABOR, ...
        return "D"
    if p.startswith("REPUBLICAN") or p == "INDEPENDENT-REPUBLICAN":   # MN GOP's name until 1995
        return "R"
    return "O"


def name_key(name) -> str:
    """Loose key for matching the same person across years: first initial + last name."""
    tokens = re.sub(r"[^A-Z ]", " ", str(name).upper()).split()
    tokens = [t for t in tokens if t not in NAME_SUFFIXES]
    if not tokens:
        return ""
    return f"{tokens[0][0]}_{tokens[-1]}"


# ---------------------------------------------------------------- race table
def build_race_table(df: pd.DataFrame) -> pd.DataFrame:
    df["major"] = df["party"].map(major_party)
    df["name_key"] = df["candidate"].map(name_key)
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0)
    if "writein" in df.columns:
        df.loc[as_bool(df["writein"]), "major"] = "O"   # write-ins never count as the major-party nominee

    race = ["year", "state_po", "district"]

    # Collapse fusion tickets: one candidate can appear on several party lines.
    cand = (
        df.groupby(race + ["name_key"], as_index=False)
        .agg(votes=("candidatevotes", "sum"),
             candidate=("candidate", "first"),
             majors=("major", lambda s: set(s)))
    )
    cand["major"] = cand["majors"].map(lambda m: "D" if "D" in m else ("R" if "R" in m else "O"))

    # Winner = candidate with the most votes in the race (any party).
    winners = cand.loc[cand.groupby(race)["votes"].idxmax(), race + ["name_key", "major"]]
    winners = winners.rename(columns={"name_key": "winner_key", "major": "winner_party"})

    def top_of(party):
        sub = cand[cand["major"] == party]
        top = sub.loc[sub.groupby(race)["votes"].idxmax(), race + ["votes", "name_key"]]
        return top.rename(columns={"votes": f"{party.lower()}_votes", "name_key": f"{party.lower()}_key"})

    races = (
        cand.groupby(race, as_index=False)["votes"].sum().rename(columns={"votes": "total_votes"})
        .merge(top_of("D"), on=race, how="left")
        .merge(top_of("R"), on=race, how="left")
        .merge(winners, on=race, how="left")
        .merge(df.groupby(race, as_index=False)["decided_by"].first(), on=race, how="left")
    )
    races[["d_votes", "r_votes"]] = races[["d_votes", "r_votes"]].fillna(0)

    two_party = races["d_votes"] + races["r_votes"]
    races["margin"] = np.where(two_party > 0, (races["d_votes"] - races["r_votes"]) / two_party, np.nan)
    races["contested"] = (races["d_votes"] > 0) & (races["r_votes"] > 0)
    races["dem_win"] = (races["winner_party"] == "D").astype(int)
    return races


# ---------------------------------------------------------------- features
def add_features(races: pd.DataFrame) -> pd.DataFrame:
    races = races.sort_values(["state_po", "district", "year"]).reset_index(drop=True)
    g = races.groupby(["state_po", "district"])

    for lag in (1, 2):
        prev_year = g["year"].shift(lag)
        valid = prev_year == races["year"] - 2 * lag      # only use truly consecutive cycles
        races[f"margin_lag{lag}"] = g["margin"].shift(lag).where(valid)
        races[f"contested_lag{lag}"] = g["contested"].shift(lag).where(valid)

    prev_winner = g["winner_key"].shift(1).where(g["year"].shift(1) == races["year"] - 2)
    races["incumbent"] = np.select(
        [prev_winner.notna() & (prev_winner == races["d_key"]),
         prev_winner.notna() & (prev_winner == races["r_key"])],
        [1, -1], default=0)                                # +1 D incumbent, -1 R incumbent, 0 open/unknown

    races["midterm"] = (races["year"] % 4 == 2).astype(int)
    races["pres_party"] = races["year"].map(PRES_PARTY)
    races["midterm_x_pres"] = races["midterm"] * races["pres_party"]   # midterm penalty term
    races["redistrict_year"] = races["year"].isin(REDISTRICT_YEARS).astype(int)
    return races


# ---------------------------------------------------------------- report
def report(raw: pd.DataFrame, races: pd.DataFrame) -> None:
    print("\nParty labels mapped to 'O' (top 15) -- check none of these should be D or R:")
    print(raw.loc[raw["major"] == "O", "party"].value_counts().head(15).to_string())

    print(f"\nRaces: {len(races):,}   years: {races['year'].min()}-{races['year'].max()}")
    by_year = races.groupby("year").agg(
        races=("dem_win", "size"),
        contested=("contested", "mean"),
        dem_wins=("dem_win", "sum"),
        third_party_wins=("winner_party", lambda s: (s == "O").sum()),
        has_lag1=("margin_lag1", lambda s: s.notna().mean()),
        incumbent_found=("incumbent", lambda s: (s != 0).mean()),
    )
    print(by_year.round(2).to_string())
    print("\nSanity checks: ~435 races per year; incumbent_found should mostly be 0.7-0.9."
          "\nNote: district numbers are reused after redistricting, so has_lag1 stays high in"
          "\nredistricting years even though those lagged margins describe different districts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", default=Path("data/processed/house_races.csv"), type=Path)
    args = ap.parse_args()

    raw = filter_general_elections(load_raw(args.input))
    races = add_features(build_race_table(raw))
    report(raw, races)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    races.drop(columns=["d_key", "r_key", "winner_key"]).to_csv(args.output, index=False)
    print(f"\nSaved {len(races):,} rows to {args.output}")


if __name__ == "__main__":
    main()
