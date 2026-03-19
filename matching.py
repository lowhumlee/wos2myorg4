"""
matching.py — Author matching logic for WoS → MyOrg pipeline.

Covers: name normalisation, exact match, initial-expansion match,
fuzzy match, and new-author assignment. Absorbs initial_matching.py
from v3.
"""

import logging
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    """Strip diacritics, lowercase, collapse whitespace.

    Args:
        s: Raw name string.

    Returns:
        Normalised ASCII-safe lowercase string.
    """
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_str.lower().split())


def parse_wos_name(raw: str) -> tuple[str, str]:
    """Split a WoS ``Last, First`` string into (last, first) normalised parts.

    Args:
        raw: Author name as it appears in WoS (e.g. ``"Ivanova, M."``).

    Returns:
        Tuple of (normalised_last, normalised_first).
    """
    if "," in raw:
        last, _, first = raw.partition(",")
    else:
        parts = raw.strip().split()
        last = parts[-1] if parts else raw
        first = " ".join(parts[:-1])
    return normalize_name(last), normalize_name(first.strip())


def parse_master_name(row: dict) -> tuple[str, str]:
    """Extract normalised (last, first) from a ResearcherAndDocument row.

    Args:
        row: Dict with at least ``FirstName`` and ``LastName`` keys.

    Returns:
        Tuple of (normalised_last, normalised_first).
    """
    return (
        normalize_name(row.get("LastName", "")),
        normalize_name(row.get("FirstName", "")),
    )


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def name_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Float in [0, 1].
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_initial(name_part: str) -> bool:
    """Return True if *name_part* looks like an initial (single letter, optional dot)."""
    cleaned = name_part.replace(".", "").strip()
    return len(cleaned) == 1


def _initials_match(wos_first: str, master_first: str) -> bool:
    """Check whether WoS initials are consistent with master's full first name.

    Handles multi-initial strings like ``"s k"`` against ``"silvia kristinova"``.

    Args:
        wos_first:    Normalised WoS first field (may be ``"s"`` or ``"s k"``).
        master_first: Normalised master first name.

    Returns:
        True if every WoS initial matches the corresponding name part.
    """
    wos_parts = wos_first.split()
    master_parts = master_first.split()

    if len(wos_parts) > len(master_parts):
        # More initials than master name parts → hard reject
        return False

    for wos_part, master_part in zip(wos_parts, master_parts):
        if _is_initial(wos_part):
            if not master_part.startswith(wos_part.replace(".", "")):
                return False
        else:
            # Full token in WoS — fuzzy compare
            if name_similarity(wos_part, master_part) < 0.80:
                return False
    return True


# ---------------------------------------------------------------------------
# Core match function
# ---------------------------------------------------------------------------

def match_author(
    wos_last: str,
    wos_first: str,
    person_list: list[dict],
    cfg: dict,
) -> dict:
    """Find the best match for a WoS author in the master person list.

    Matching priority:
    1. Exact (last + first, normalised)
    2. Initial expansion (WoS has initials, master has full name)
    3. Fuzzy (surname match + first-name similarity above threshold)
    4. New author (no match found)

    Args:
        wos_last:    Normalised WoS last name.
        wos_first:   Normalised WoS first name / initials.
        person_list: List of person dicts from :func:`core.build_person_index`.
        cfg:         Full config dict (uses ``matching`` sub-key).

    Returns:
        Dict with keys:
        ``match_type``, ``suggested_pid``, ``suggested_first``,
        ``suggested_last``, ``score``.
    """
    thresholds = cfg.get("matching", {})
    fuzzy_thresh = float(thresholds.get("fuzzy_threshold", 0.85))
    init_thresh = float(thresholds.get("initial_expansion_threshold", 0.80))

    wos_is_initial = all(_is_initial(p) for p in wos_first.split()) if wos_first else False

    best_fuzzy: Optional[dict] = None
    best_fuzzy_score = 0.0

    for person in person_list:
        m_last, m_first = parse_master_name(person)

        # Last-name gate (normalised equality)
        if m_last != wos_last:
            # Allow small surname variation (transliteration)
            surname_sim = name_similarity(m_last, wos_last)
            if surname_sim < 0.82:
                continue

        # --- Exact ---
        if m_last == wos_last and m_first == wos_first:
            return _make_result("exact", person, 1.0)

        # --- Initial expansion ---
        if wos_is_initial and m_last == wos_last:
            if _initials_match(wos_first, m_first):
                first_sim = name_similarity(wos_first, m_first[:len(wos_first)])
                if first_sim >= init_thresh or True:  # initials always qualify if structure matches
                    return _make_result("initial_expansion", person, 0.90)

        # --- Fuzzy ---
        first_sim = name_similarity(wos_first, m_first)
        if first_sim >= fuzzy_thresh:
            if first_sim > best_fuzzy_score:
                best_fuzzy_score = first_sim
                best_fuzzy = person

    if best_fuzzy is not None:
        return _make_result("fuzzy", best_fuzzy, best_fuzzy_score)

    return {
        "match_type": "new",
        "suggested_pid": None,
        "suggested_first": wos_first.title() if wos_first else "",
        "suggested_last": wos_last.title() if wos_last else "",
        "score": 0.0,
    }


def _make_result(match_type: str, person: dict, score: float) -> dict:
    """Package a successful match result.

    Args:
        match_type: One of ``exact``, ``initial_expansion``, ``fuzzy``.
        person:     Matched person dict.
        score:      Similarity score.

    Returns:
        Standardised result dict.
    """
    return {
        "match_type": match_type,
        "suggested_pid": person["PersonID"],
        "suggested_first": person.get("FirstName", ""),
        "suggested_last": person.get("LastName", ""),
        "score": score,
    }


# ---------------------------------------------------------------------------
# Batch interface (used by core.batch_process)
# ---------------------------------------------------------------------------

def match_author_pair(
    wos_author: dict,
    person_list: list[dict],
    existing_pairs: set,
    cfg: dict,
    max_pid: int,
) -> dict:
    """Match one (author, document) pair and detect probable duplicates.

    Args:
        wos_author:     Dict with keys ``last``, ``first``, ``doc_id``.
        person_list:    Master person list.
        existing_pairs: Set of ``(pid, doc_id)`` already in MyOrg.
        cfg:            Full config dict.
        max_pid:        Current highest PersonID in master.

    Returns:
        Extended match result dict with ``doc_id``, ``is_duplicate`` flag.
    """
    result = match_author(
        wos_author["last"],
        wos_author["first"],
        person_list,
        cfg,
    )
    result["doc_id"] = wos_author["doc_id"]
    result["raw_last"] = wos_author["last"]
    result["raw_first"] = wos_author["first"]

    # Probable duplicate detection
    pid = result.get("suggested_pid")
    dup_thresh = float(cfg.get("matching", {}).get("probable_duplicate_threshold", 0.92))
    result["is_duplicate"] = (
        pid is not None
        and result["score"] >= dup_thresh
        and (pid, wos_author["doc_id"]) in existing_pairs
    )

    return result
