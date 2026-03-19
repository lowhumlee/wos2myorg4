"""
core.py — Parsing, MUV extraction, and batch processing.

JSON structure confirmed from export.json:
  REC → UID
  REC → static_data → summary → names → name[]
  REC → static_data → fullrecord_metadata → addresses → address_name[]
    → address_spec → addr_no (int), organizations → organization[], city, country
"""

import csv
import io
import json
import logging
import re
from typing import Optional

import pandas as pd

from matching import match_author_pair, normalize_name, parse_wos_name

logger = logging.getLogger(__name__)

_STRUCTURE_LOGGED = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Master person index
# ---------------------------------------------------------------------------

def build_person_index(csv_bytes: bytes) -> tuple[list[dict], int, set]:
    """Parse ResearcherAndDocument.csv → (person_list, max_pid, existing_pairs).

    Each person dict carries:
      PersonID, FirstName, LastName,
      OrganizationID  – first/primary org ID (backward compat)
      OrganizationIDs – deduplicated list of ALL org IDs for this person
    """
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
    required = {"PersonID", "FirstName", "LastName", "DocumentID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ResearcherAndDocument.csv missing columns: {missing}")

    existing_pairs: set = set()
    max_pid = 0
    seen_pids: dict[str, dict] = {}

    for _, row in df.iterrows():
        pid_str = row["PersonID"].strip()
        doc_id  = row["DocumentID"].strip()
        oid     = row.get("OrganizationID", "").strip()
        try:
            pid_int = int(pid_str)
        except ValueError:
            logger.warning("Non-integer PersonID skipped: %s", pid_str)
            continue

        max_pid = max(max_pid, pid_int)
        existing_pairs.add((pid_int, doc_id))

        if pid_str not in seen_pids:
            seen_pids[pid_str] = {
                "PersonID":       pid_int,
                "FirstName":      row.get("FirstName", "").strip(),
                "LastName":       row.get("LastName", "").strip(),
                "OrganizationID":  oid,
                "OrganizationIDs": [oid] if oid else [],
            }
        else:
            # Accumulate additional org IDs for the same person
            if oid and oid not in seen_pids[pid_str]["OrganizationIDs"]:
                seen_pids[pid_str]["OrganizationIDs"].append(oid)

    person_list = list(seen_pids.values())
    logger.info(
        "Person index: %d unique persons, %d existing pairs, max PID=%d",
        len(person_list), len(existing_pairs), max_pid,
    )
    return person_list, max_pid, existing_pairs


def parse_org_hierarchy(csv_bytes: bytes) -> list[dict]:
    """Parse OrgHierarchy.csv → list of org dicts."""
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# MUV affiliation detection
# ---------------------------------------------------------------------------

def _is_muv_affiliation(norm_affil: str, cfg: dict) -> bool:
    for pat in cfg.get("affiliation_patterns", []):
        if pat in norm_affil:
            return True
    for sc in cfg.get("special_cases", []):
        contains = sc.get("contains", [])
        excludes = sc.get("excludes", [])
        if sc.get("mode", "all") == "all":
            if all(c in norm_affil for c in contains):
                if not any(e in norm_affil for e in excludes):
                    return True
    return False


def _norm(s: str) -> str:
    return normalize_name(s)


def _ensure_list(val) -> list:
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


# ---------------------------------------------------------------------------
# WoS JSON parsing  (structure confirmed from export.json)
# ---------------------------------------------------------------------------

def _log_structure_once(raw: dict):
    global _STRUCTURE_LOGGED
    if _STRUCTURE_LOGGED:
        return
    _STRUCTURE_LOGGED = True

    def _peek(obj, depth=0):
        if depth >= 3:
            return "..."
        if isinstance(obj, dict):
            return {k: _peek(v, depth + 1) for k, v in list(obj.items())[:5]}
        if isinstance(obj, list):
            return [_peek(obj[0], depth + 1)] if obj else []
        return repr(obj)[:60]

    logger.info("=== WoS JSON record structure ===\n%s", json.dumps(_peek(raw), indent=2))


def parse_wos_json_record(raw: dict) -> Optional[dict]:
    """Parse one WoS JSON REC into internal format.

    Returns dict with:
      UT        – WoS document ID
      AU_list   – list of author dicts (last, first, display, addr_nos, orcid)
      addr_map  – {addr_no: normalised_affil_string}
      addr_raw  – {addr_no: raw full_address string} for display in UI
    """
    _log_structure_once(raw)

    try:
        ut: str = (raw.get("UID") or "").strip()
        if not ut:
            return None

        sd = raw.get("static_data") or {}

        # ── Authors ──────────────────────────────────────────────────────────
        names_node = (sd.get("summary") or {}).get("names") or {}
        name_items = _ensure_list(names_node.get("name"))

        authors = []
        for n in name_items:
            if (n.get("role") or "").lower() not in ("author", ""):
                continue

            addr_nos_raw = n.get("addr_no") or ""
            addr_nos = [int(x) for x in str(addr_nos_raw).split() if x.isdigit()]

            last    = _norm(n.get("last_name") or "")
            first   = _norm(n.get("first_name") or "")
            display = n.get("display_name") or n.get("full_name") or f"{last}, {first}"

            orcid = ""
            for item in _ensure_list((n.get("data-item-ids") or {}).get("data-item-id")):
                if isinstance(item, dict) and "ORCID" in (item.get("id-type") or ""):
                    orcid = item.get("content") or ""
                    break

            if not last:
                continue

            authors.append({
                "display":  display,
                "last":     last,
                "first":    first,
                "addr_nos": addr_nos,
                "orcid":    orcid,
            })

        # ── Addresses ────────────────────────────────────────────────────────
        fm = sd.get("fullrecord_metadata") or {}
        addr_names = _ensure_list((fm.get("addresses") or {}).get("address_name"))

        addr_map: dict[int, str] = {}   # normalised, for MUV detection
        addr_raw: dict[int, str] = {}   # original full_address, for UI display

        for addr in addr_names:
            spec = addr.get("address_spec") or {}
            try:
                addr_no = int(spec.get("addr_no") or -1)
            except (ValueError, TypeError):
                continue

            parts: list[str] = []
            org_items = _ensure_list((spec.get("organizations") or {}).get("organization"))
            for org in org_items:
                text = (org.get("content") or "") if isinstance(org, dict) else str(org)
                if text:
                    parts.append(text)

            parts.append(spec.get("city") or "")
            parts.append(spec.get("country") or "")

            if addr_no >= 0:
                addr_map[addr_no] = _norm(" ".join(p for p in parts if p))
                # Keep the raw full_address for display; fall back to joined parts
                addr_raw[addr_no] = spec.get("full_address") or ", ".join(p for p in parts if p)

        return {"UT": ut, "AU_list": authors, "addr_map": addr_map, "addr_raw": addr_raw}

    except Exception as exc:
        logger.error("Failed to parse JSON record: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# WoS CSV parsing
# ---------------------------------------------------------------------------

def parse_wos_csv(csv_bytes: bytes) -> list[dict]:
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

        au_raw = [a.strip() for a in (row.get("AU") or "").split(";") if a.strip()]
        af_raw = [a.strip() for a in (row.get("AF") or "").split(";") if a.strip()]
        c1_raw = (row.get("C1") or "").strip()
        c3_raw = (row.get("C3") or "").strip()

        authors = []
        for idx, au in enumerate(au_raw):
            last, first = parse_wos_name(au)
            display = af_raw[idx] if idx < len(af_raw) else au
            authors.append({
                "display":      display,
                "last":         last,
                "first":        first,
                "addr_nos":     [],
                "orcid":        "",
                "c1_fragment":  "",
            })

        if c1_raw:
            for au_group, affil in re.findall(r"\[([^\]]+)\]\s*([^[]+)", c1_raw):
                affil_norm = _norm(affil)
                for au_name in au_group.split(";"):
                    au_last, _ = parse_wos_name(au_name.strip())
                    for auth in authors:
                        if auth["last"] == au_last:
                            auth["c1_fragment"] = affil_norm

        records.append({
            "UT":      ut,
            "AU_list": authors,
            "addr_map": {},
            "addr_raw": {},
            "C3":      _norm(c3_raw),
        })

    logger.info("Parsed %d records from WoS CSV", len(records))
    return records


# ---------------------------------------------------------------------------
# MUV author extraction
# ---------------------------------------------------------------------------

def extract_muv_author_pairs(records: list[dict], cfg: dict) -> list[dict]:
    """Filter records to MUV-affiliated (author, document) pairs.

    Each pair dict carries:
      last, first, display, orcid, doc_id, affil (normalised),
      affil_display (raw, for UI), source
    """
    hospital_patterns = [_norm(p) for p in cfg.get("hospital_partners", [])]
    pairs: list[dict] = []
    skipped = 0

    for rec in records:
        ut       = rec["UT"]
        au_list  = rec.get("AU_list", [])
        addr_map = rec.get("addr_map", {})
        addr_raw = rec.get("addr_raw", {})

        if not au_list:
            skipped += 1
            continue

        muv_pairs: list[dict] = []
        has_any_muv = False

        for auth in au_list:
            affil      = _resolve_affiliation(auth, addr_map)
            affil_disp = _resolve_affiliation_raw(auth, addr_raw)
            if _is_muv_affiliation(affil, cfg):
                has_any_muv = True
                muv_pairs.append(_make_pair(auth, ut, affil, affil_disp, "direct"))

        if not has_any_muv:
            for auth in au_list:
                affil      = auth.get("c1_fragment", "") or _resolve_affiliation(auth, addr_map)
                affil_disp = _resolve_affiliation_raw(auth, addr_raw)
                if "varna" in affil and not _already_in(auth, ut, muv_pairs):
                    muv_pairs.append(_make_pair(auth, ut, affil, affil_disp, "c3_tier1"))

        if has_any_muv:
            for auth in au_list:
                affil      = _resolve_affiliation(auth, addr_map)
                affil_disp = _resolve_affiliation_raw(auth, addr_raw)
                if any(hp in affil for hp in hospital_patterns) and not _already_in(auth, ut, muv_pairs):
                    muv_pairs.append(_make_pair(auth, ut, affil, affil_disp, "c3_tier2"))

        if not muv_pairs:
            skipped += 1

        pairs.extend(muv_pairs)

    logger.info("Extracted %d MUV author-document pairs (%d UTs skipped)", len(pairs), skipped)
    return pairs


def _resolve_affiliation(auth: dict, addr_map: dict[int, str]) -> str:
    if auth.get("addr_nos"):
        return " ".join(addr_map.get(no, "") for no in auth["addr_nos"] if addr_map.get(no))
    return auth.get("c1_fragment", "")


def _resolve_affiliation_raw(auth: dict, addr_raw: dict[int, str]) -> str:
    """Return the human-readable affiliation string for UI display."""
    if auth.get("addr_nos"):
        parts = [addr_raw.get(no, "") for no in auth["addr_nos"] if addr_raw.get(no)]
        return " | ".join(dict.fromkeys(parts))  # deduplicated, order-preserved
    return auth.get("c1_fragment", "")


def _already_in(auth: dict, ut: str, pairs: list[dict]) -> bool:
    return any(p["display"] == auth["display"] and p["doc_id"] == ut for p in pairs)


def _make_pair(auth: dict, doc_id: str, affil: str, affil_display: str, source: str) -> dict:
    return {
        "last":          auth["last"],
        "first":         auth["first"],
        "display":       auth["display"],
        "orcid":         auth.get("orcid", ""),
        "doc_id":        doc_id,
        "affil":         affil,
        "affil_display": affil_display,
        "source":        source,
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
    confirmed, needs_review, already_uploaded, new_persons = [], [], [], []

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
        len(confirmed), len(needs_review), len(already_uploaded), len(new_persons),
    )
    return {
        "confirmed":       confirmed,
        "needs_review":    needs_review,
        "already_uploaded": already_uploaded,
        "new_persons":     new_persons,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_by_ut(pairs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for p in pairs:
        grouped.setdefault(p["doc_id"], []).append(p)
    return grouped


def build_upload_row(pid: int, first: str, last: str, doc_id: str, org_id: str) -> dict:
    return {
        "PersonID":       pid,
        "FirstName":      first,
        "LastName":       last,
        "OrganizationID": org_id,
        "DocumentID":     doc_id,
    }


def deduplicate_upload_rows(rows: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for row in rows:
        key = (row["PersonID"], row["DocumentID"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out
