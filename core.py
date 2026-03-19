"""
core.py — Parsing, MUV extraction, and batch processing.

Handles both WoS JSON (from api.py) and WoS CSV (user upload).
All matching is delegated to matching.py.
"""

import csv
import io
import logging
import unicodedata
from typing import Optional

import pandas as pd

from matching import match_author_pair, normalize_name, parse_wos_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    """Load and return config.json as a dict.

    Args:
        path: Path to config file.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Master person index (ResearcherAndDocument.csv)
# ---------------------------------------------------------------------------

def build_person_index(csv_bytes: bytes) -> tuple[list[dict], int, set]:
    """Parse ResearcherAndDocument.csv into a usable index.

    Args:
        csv_bytes: Raw bytes of the CSV file.

    Returns:
        Tuple of:
        - ``person_list``: List of person dicts (PersonID, FirstName, LastName, …).
        - ``max_pid``:     Highest numeric PersonID present.
        - ``existing_pairs``: Set of ``(PersonID, DocumentID)`` already in MyOrg.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
    required = {"PersonID", "FirstName", "LastName", "DocumentID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ResearcherAndDocument.csv missing columns: {missing}")

    person_list: list[dict] = []
    existing_pairs: set = set()
    max_pid = 0

    seen_pids: dict[str, dict] = {}

    for _, row in df.iterrows():
        pid_str = row["PersonID"].strip()
        doc_id = row["DocumentID"].strip()

        try:
            pid_int = int(pid_str)
        except ValueError:
            logger.warning("Non-integer PersonID skipped: %s", pid_str)
            continue

        max_pid = max(max_pid, pid_int)
        existing_pairs.add((pid_int, doc_id))

        if pid_str not in seen_pids:
            seen_pids[pid_str] = {
                "PersonID": pid_int,
                "FirstName": row.get("FirstName", "").strip(),
                "LastName": row.get("LastName", "").strip(),
                "OrganizationID": row.get("OrganizationID", "").strip(),
            }

    person_list = list(seen_pids.values())
    logger.info(
        "Person index: %d unique persons, %d existing pairs, max PID=%d",
        len(person_list),
        len(existing_pairs),
        max_pid,
    )
    return person_list, max_pid, existing_pairs


def parse_org_hierarchy(csv_bytes: bytes) -> list[dict]:
    """Parse OrgHierarchy.csv into a list of org dicts (optional).

    Args:
        csv_bytes: Raw bytes of the CSV file.

    Returns:
        List of dicts with org info.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# MUV affiliation detection
# ---------------------------------------------------------------------------

def _is_muv_affiliation(norm_affil: str, cfg: dict) -> bool:
    """Return True if a normalised affiliation string belongs to MUV.

    Args:
        norm_affil: Normalised (lowercase, ASCII) affiliation string.
        cfg:        Full config dict.

    Returns:
        True if the affiliation matches MUV patterns.
    """
    patterns: list[str] = cfg.get("affiliation_patterns", [])
    for pat in patterns:
        if pat in norm_affil:
            return True

    # Special cases
    for sc in cfg.get("special_cases", []):
        contains: list[str] = sc.get("contains", [])
        excludes: list[str] = sc.get("excludes", [])
        mode: str = sc.get("mode", "all")
        if mode == "all":
            if all(c in norm_affil for c in contains):
                if not any(e in norm_affil for e in excludes):
                    return True

    return False


def _norm(s: str) -> str:
    """Convenience alias for full normalisation."""
    return normalize_name(s)


# ---------------------------------------------------------------------------
# WoS JSON parsing
# ---------------------------------------------------------------------------

def parse_wos_json_record(raw: dict) -> Optional[dict]:
    """Parse one raw WoS JSON record into a normalised internal dict.

    Args:
        raw: Raw record dict from api.py.

    Returns:
        Parsed dict with keys ``UT``, ``AU_list``, ``addr_map``, or None on failure.
    """
    try:
        ut: str = raw.get("UID", "").strip()
        if not ut:
            logger.warning("Record missing UID, skipping")
            return None

        sd = raw.get("static_data", {})
        summary = sd.get("summary", {})

        # --- Authors ---
        names_node = summary.get("names", {})
        name_items = names_node.get("name", [])
        if isinstance(name_items, dict):
            name_items = [name_items]

        authors = []
        for n in name_items:
            if n.get("role", "").lower() != "author":
                continue
            addr_nos_raw = n.get("addr_no", "")
            addr_nos = (
                [int(x) for x in str(addr_nos_raw).split() if x.isdigit()]
                if addr_nos_raw
                else []
            )
            authors.append(
                {
                    "display": n.get("full_name", n.get("display_name", "")),
                    "last": _norm(n.get("last_name", "")),
                    "first": _norm(n.get("first_name", "")),
                    "addr_nos": addr_nos,
                    "orcid": n.get("orcid_id", ""),
                }
            )

        # --- Addresses ---
        fm = raw.get("fullrecord_metadata", {})
        addr_names = fm.get("addresses", {}).get("address_name", [])
        if isinstance(addr_names, dict):
            addr_names = [addr_names]

        addr_map: dict[int, str] = {}
        for addr in addr_names:
            spec = addr.get("address_spec", {})
            try:
                addr_no = int(spec.get("addr_no", -1))
            except (ValueError, TypeError):
                continue
            orgs = spec.get("organizations", {}).get("organization", [])
            if isinstance(orgs, dict):
                orgs = [orgs]
            org_names = []
            for org in orgs:
                content = org.get("content", "") if isinstance(org, dict) else str(org)
                if content:
                    org_names.append(content)
            full_affil = " ".join(org_names) + " " + spec.get("city", "")
            addr_map[addr_no] = _norm(full_affil)

        return {"UT": ut, "AU_list": authors, "addr_map": addr_map}

    except Exception as exc:
        logger.error("Failed to parse JSON record: %s", exc)
        return None


# ---------------------------------------------------------------------------
# WoS CSV parsing
# ---------------------------------------------------------------------------

def parse_wos_csv(csv_bytes: bytes) -> list[dict]:
    """Parse a WoS plain-text CSV export into internal record dicts.

    Args:
        csv_bytes: Raw bytes of the WoS export CSV.

    Returns:
        List of dicts with keys ``UT``, ``AU_list``, ``addr_map``.
    """
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = csv_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    records = []

    for row in reader:
        ut = (row.get("UT") or "").strip()
        if not ut:
            continue

        # AU / AF / C1 columns
        au_raw: list[str] = [a.strip() for a in (row.get("AU") or "").split(";") if a.strip()]
        af_raw: list[str] = [a.strip() for a in (row.get("AF") or "").split(";") if a.strip()]
        c1_raw: str = (row.get("C1") or "").strip()
        c3_raw: str = (row.get("C3") or "").strip()

        authors = []
        for idx, au in enumerate(au_raw):
            last, first = parse_wos_name(au)
            full_name = af_raw[idx] if idx < len(af_raw) else au
            authors.append(
                {
                    "display": full_name,
                    "last": last,
                    "first": first,
                    "addr_nos": [],  # CSV has no addr_no linkage
                    "orcid": "",
                    "c1_fragment": "",  # filled below
                }
            )

        # Build a simple addr_map from C1 blocks
        addr_map: dict[int, str] = {}
        if c1_raw:
            # C1 format: "[Author A; Author B] Affiliation text. [Author C] Other affil."
            import re
            blocks = re.findall(r"\[([^\]]+)\]\s*([^[]+)", c1_raw)
            for au_group, affil in blocks:
                affil_norm = _norm(affil)
                for au_name in au_group.split(";"):
                    au_name = au_name.strip()
                    au_last, _ = parse_wos_name(au_name)
                    for auth in authors:
                        if auth["last"] == au_last:
                            auth["c1_fragment"] = affil_norm

        records.append(
            {"UT": ut, "AU_list": authors, "addr_map": addr_map, "C3": _norm(c3_raw)}
        )

    logger.info("Parsed %d records from WoS CSV", len(records))
    return records


# ---------------------------------------------------------------------------
# MUV author extraction
# ---------------------------------------------------------------------------

def extract_muv_author_pairs(records: list[dict], cfg: dict) -> list[dict]:
    """Filter records to (author, document) pairs affiliated with MUV.

    Implements two-tier C3 fallback for CSV records:
    - Tier 1: No MUV found in C1 → add authors whose C1 fragment contains "varna".
    - Tier 2: Some MUV in C1 → add hospital partners linked to the same paper.

    Args:
        records: Output of :func:`parse_wos_csv` or JSON records list.
        cfg:     Full config dict.

    Returns:
        List of author-document pair dicts:
        ``{last, first, display, doc_id, affil, source}``.
    """
    hospital_patterns: list[str] = [_norm(p) for p in cfg.get("hospital_partners", [])]
    pairs: list[dict] = []
    skipped = 0

    for rec in records:
        ut = rec["UT"]
        au_list = rec.get("AU_list", [])
        addr_map = rec.get("addr_map", {})
        c3 = rec.get("C3", "")

        if not au_list:
            logger.debug("UT %s: no authors, skipping", ut)
            skipped += 1
            continue

        muv_pairs_this_ut: list[dict] = []
        has_any_muv = False

        for auth in au_list:
            # Resolve affiliation string
            affil_str = _resolve_affiliation(auth, addr_map)

            if _is_muv_affiliation(affil_str, cfg):
                has_any_muv = True
                muv_pairs_this_ut.append(_make_pair(auth, ut, affil_str, "direct"))

        # C3 Tier 1: no MUV at all in C1/JSON → fallback to varna keyword
        if not has_any_muv:
            for auth in au_list:
                affil_str = _resolve_affiliation(auth, addr_map)
                if "varna" in affil_str and not any(
                    p["display"] == auth["display"] and p["doc_id"] == ut
                    for p in muv_pairs_this_ut
                ):
                    muv_pairs_this_ut.append(_make_pair(auth, ut, affil_str, "c3_tier1"))

        # C3 Tier 2: some MUV → also add hospital partners
        if has_any_muv:
            for auth in au_list:
                affil_str = _resolve_affiliation(auth, addr_map)
                if any(hp in affil_str for hp in hospital_patterns):
                    if not any(
                        p["display"] == auth["display"] and p["doc_id"] == ut
                        for p in muv_pairs_this_ut
                    ):
                        muv_pairs_this_ut.append(_make_pair(auth, ut, affil_str, "c3_tier2"))

        if not muv_pairs_this_ut:
            logger.debug("UT %s: no MUV authors found", ut)
            skipped += 1

        pairs.extend(muv_pairs_this_ut)

    logger.info(
        "Extracted %d MUV author-document pairs (%d UTs skipped)", len(pairs), skipped
    )
    return pairs


def _resolve_affiliation(auth: dict, addr_map: dict[int, str]) -> str:
    """Get normalised affiliation string for an author.

    Prefers JSON addr_no linkage; falls back to c1_fragment from CSV parsing.
    """
    if auth.get("addr_nos"):
        parts = [addr_map.get(no, "") for no in auth["addr_nos"]]
        return " ".join(p for p in parts if p)
    return auth.get("c1_fragment", "")


def _make_pair(auth: dict, doc_id: str, affil: str, source: str) -> dict:
    return {
        "last": auth["last"],
        "first": auth["first"],
        "display": auth["display"],
        "orcid": auth.get("orcid", ""),
        "doc_id": doc_id,
        "affil": affil,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_process(
    pairs: list[dict],
    person_list: list[dict],
    existing_pairs: set,
    cfg: dict,
    max_pid: int,
) -> dict:
    """Match all author-document pairs against the master roster.

    Args:
        pairs:          Output of :func:`extract_muv_author_pairs`.
        person_list:    From :func:`build_person_index`.
        existing_pairs: From :func:`build_person_index`.
        cfg:            Full config dict.
        max_pid:        Highest existing PersonID.

    Returns:
        Dict with keys:
        ``confirmed`` (auto-matched), ``needs_review``, ``already_uploaded``,
        ``new_persons`` (authors needing new PIDs).
        Each value is a list of result dicts.
    """
    confirmed: list[dict] = []
    needs_review: list[dict] = []
    already_uploaded: list[dict] = []
    new_persons: list[dict] = []

    for pair in pairs:
        result = match_author_pair(pair, person_list, existing_pairs, cfg, max_pid)

        if result["is_duplicate"]:
            already_uploaded.append(result)
            continue

        mt = result["match_type"]
        if mt == "exact":
            confirmed.append(result)
        elif mt in ("initial_expansion", "fuzzy"):
            needs_review.append(result)
        elif mt == "new":
            new_persons.append(result)
            needs_review.append(result)

    logger.info(
        "Batch: %d confirmed, %d needs review, %d already uploaded, %d new",
        len(confirmed),
        len(needs_review),
        len(already_uploaded),
        len(new_persons),
    )
    return {
        "confirmed": confirmed,
        "needs_review": needs_review,
        "already_uploaded": already_uploaded,
        "new_persons": new_persons,
    }


# ---------------------------------------------------------------------------
# UT grouping helper
# ---------------------------------------------------------------------------

def group_by_ut(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group author-document pairs by UT (document ID).

    Args:
        pairs: List of pair dicts (each has ``doc_id``).

    Returns:
        Ordered dict mapping UT → list of pairs for that UT.
    """
    grouped: dict[str, list[dict]] = {}
    for p in pairs:
        grouped.setdefault(p["doc_id"], []).append(p)
    return grouped


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def build_upload_row(
    pid: int,
    first: str,
    last: str,
    doc_id: str,
    org_id: str,
) -> dict:
    """Build one row for upload.csv.

    Args:
        pid:    PersonID (int).
        first:  FirstName.
        last:   LastName.
        doc_id: WoS DocumentID (UT).
        org_id: OrganizationID from config.

    Returns:
        Dict with the five required columns.
    """
    return {
        "PersonID": pid,
        "FirstName": first,
        "LastName": last,
        "OrganizationID": org_id,
        "DocumentID": doc_id,
    }


def deduplicate_upload_rows(rows: list[dict]) -> list[dict]:
    """Remove duplicate (PersonID, DocumentID) rows.

    Args:
        rows: List of upload row dicts.

    Returns:
        Deduplicated list preserving first occurrence.
    """
    seen: set = set()
    out: list[dict] = []
    for row in rows:
        key = (row["PersonID"], row["DocumentID"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out
