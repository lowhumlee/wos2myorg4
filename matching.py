"""
matching.py — Author matching logic for WoS → MyOrg pipeline.

Returns match results that include suggested_org_ids from the matched person.
"""

import logging
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_name(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_str.lower().split())


def parse_wos_name(raw: str) -> tuple[str, str]:
    if "," in raw:
        last, _, first = raw.partition(",")
    else:
        parts = raw.strip().split()
        last  = parts[-1] if parts else raw
        first = " ".join(parts[:-1])
    return normalize_name(last), normalize_name(first.strip())


def parse_master_name(row: dict) -> tuple[str, str]:
    return (
        normalize_name(row.get("LastName", "")),
        normalize_name(row.get("FirstName", "")),
    )


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_initial(name_part: str) -> bool:
    return len(name_part.replace(".", "").strip()) == 1


def _initials_match(wos_first: str, master_first: str) -> bool:
    wos_parts    = wos_first.split()
    master_parts = master_first.split()
    if len(wos_parts) > len(master_parts):
        return False
    for wos_part, master_part in zip(wos_parts, master_parts):
        if _is_initial(wos_part):
            if not master_part.startswith(wos_part.replace(".", "")):
                return False
        else:
            if name_similarity(wos_part, master_part) < 0.80:
                return False
    return True


def _person_org_ids(person: dict) -> list[str]:
    """Return the deduplicated list of OrganizationIDs for a person."""
    ids = person.get("OrganizationIDs") or []
    if not ids and person.get("OrganizationID"):
        ids = [person["OrganizationID"]]
    return ids


def match_author(
    wos_last: str,
    wos_first: str,
    person_list: list[dict],
    cfg: dict,
) -> dict:
    """Find the best match for a WoS author in the master person list.

    Returns dict with:
      match_type, suggested_pid, suggested_first, suggested_last,
      suggested_org_ids, score
    """
    thresholds   = cfg.get("matching", {})
    fuzzy_thresh = float(thresholds.get("fuzzy_threshold", 0.85))
    init_thresh  = float(thresholds.get("initial_expansion_threshold", 0.80))

    wos_is_initial = all(_is_initial(p) for p in wos_first.split()) if wos_first else False

    best_fuzzy: Optional[dict] = None
    best_fuzzy_score = 0.0

    for person in person_list:
        m_last, m_first = parse_master_name(person)

        if m_last != wos_last:
            if name_similarity(m_last, wos_last) < 0.82:
                continue

        # Exact
        if m_last == wos_last and m_first == wos_first:
            return _make_result("exact", person, 1.0)

        # Initial expansion
        if wos_is_initial and m_last == wos_last:
            if _initials_match(wos_first, m_first):
                return _make_result("initial_expansion", person, 0.90)

        # Fuzzy — also catch "eleonora" vs "eleonora g" (one is prefix of the other)
        first_sim = name_similarity(wos_first, m_first)
        prefix_match = (
            wos_first and m_first and (
                m_first.startswith(wos_first) or wos_first.startswith(m_first)
            )
        )
        effective_sim = max(first_sim, 0.87 if prefix_match else 0.0)
        if effective_sim >= fuzzy_thresh and effective_sim > best_fuzzy_score:
            best_fuzzy_score = effective_sim
            best_fuzzy = person

    if best_fuzzy is not None:
        return _make_result("fuzzy", best_fuzzy, best_fuzzy_score)

    return {
        "match_type":        "new",
        "suggested_pid":     None,
        "suggested_first":   wos_first.title() if wos_first else "",
        "suggested_last":    wos_last.title() if wos_last else "",
        "suggested_org_ids": [],
        "score":             0.0,
    }


def _make_result(match_type: str, person: dict, score: float) -> dict:
    return {
        "match_type":        match_type,
        "suggested_pid":     person["PersonID"],
        "suggested_first":   person.get("FirstName", ""),
        "suggested_last":    person.get("LastName", ""),
        "suggested_org_ids": _person_org_ids(person),
        "score":             score,
    }


def match_author_pair(
    wos_author: dict,
    person_list: list[dict],
    existing_pairs: set,
    cfg: dict,
    max_pid: int,
) -> dict:
    """Match one (author, document) pair and detect probable duplicates."""
    result = match_author(
        wos_author["last"],
        wos_author["first"],
        person_list,
        cfg,
    )
    result["doc_id"]        = wos_author["doc_id"]
    result["raw_last"]      = wos_author["last"]
    result["raw_first"]     = wos_author["first"]
    result["affil_display"] = wos_author.get("affil_display", "")

    pid       = result.get("suggested_pid")
    dup_thresh = float(cfg.get("matching", {}).get("probable_duplicate_threshold", 0.92))
    result["is_duplicate"] = (
        pid is not None
        and result["score"] >= dup_thresh
        and (pid, wos_author["doc_id"]) in existing_pairs
    )

    return result
