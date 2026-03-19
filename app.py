"""
app.py — Streamlit UI for WoS → MyOrg pipeline (v4).

Three tabs:
  1. Load Data  — API date-range query OR CSV upload
  2. Review     — UT-centric author decisions
  3. Export     — download upload.csv + full output

OrganizationID logic:
  - Existing authors  → org IDs inherited from ResearcherAndDocument.csv roster
  - New authors       → user must supply org ID; API affiliation shown as hint
"""

import io
import logging
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from api import WoSClient, WoSAPIError, validate_api_key
from core import (
    load_config,
    build_person_index,
    parse_org_hierarchy,
    parse_wos_csv,
    parse_wos_json_record,
    extract_muv_author_pairs,
    batch_process,
    group_by_ut,
    build_upload_row,
    deduplicate_upload_rows,
)
from matching import normalize_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
cfg      = load_config(CFG_PATH)
ORG_ID   = cfg.get("organization_id", "MED_UNIV_VARNA")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
KEYS_DEFAULTS = {
    "confirmed_rows":       [],
    "skipped_rows":         [],
    "ut_locked":            {},
    "author_decs":          {},   # key → decision dict
    "existing_pairs":       set(),
    "max_pid":              0,
    "staging_pid_counter":  None,
    "person_list":          [],
    "all_pairs":            [],
    "batch_result":         None,
    "ut_list":              [],
    "source":               None,
    "orgs":                 [],
    "org_map":              {},
}

for k, v in KEYS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_staging_pid() -> int:
    """Return next available staging PersonID (sequentially above max_pid)."""
    if not isinstance(st.session_state.staging_pid_counter, int):
        st.session_state.staging_pid_counter = st.session_state.max_pid + 1
    else:
        st.session_state.staging_pid_counter += 1
    return st.session_state.staging_pid_counter


def _dec_key(pair: dict) -> str:
    # Handles both pair dicts (last/first) and match result dicts (raw_last/raw_first)
    last  = pair.get("last")  or pair.get("raw_last",  "")
    first = pair.get("first") or pair.get("raw_first", "")
    return f"{last}|{first}|{pair['doc_id']}"


def _reset_session_data():
    for k, v in KEYS_DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, (list, dict, set)) else v
    st.session_state.pop("_api_records", None)
    st.session_state.pop("_api_key", None)


def _store_confirmed_ut(ut: str, decisions: dict[str, dict]):
    """Lock a UT and write its rows into confirmed_rows / skipped_rows."""
    if st.session_state.ut_locked.get(ut):
        return

    new_confirmed: list[dict] = []
    new_skipped:   list[dict] = []
    seen: set = set()

    for key, dec in decisions.items():
        action = dec.get("action")

        if action == "skip":
            new_skipped.append({
                "display": dec.get("display", ""),
                "doc_id":  ut,
                "Status":  "2SKIP",
                "Note":    dec.get("note", "User skipped"),
            })
            continue

        pid    = dec.get("pid")
        first  = dec.get("first", "")
        last   = dec.get("last", "")
        doc_id = dec.get("doc_id", ut)
        is_new = dec.get("is_new", False)

        # One row per org ID (existing authors may have multiple)
        org_ids = dec.get("org_ids") or [dec.get("org_id", "")]
        if not org_ids:
            org_ids = [""]

        for oid in org_ids:
            k = (pid, doc_id, oid)
            if k in seen:
                continue
            seen.add(k)
            row = build_upload_row(pid, first, last, doc_id, oid)
            row["Status"]  = "4UP"
            row["Note"]    = dec.get("note", "")
            row["is_new"]  = is_new
            new_confirmed.append(row)

    st.session_state.confirmed_rows.extend(new_confirmed)
    st.session_state.skipped_rows.extend(new_skipped)
    st.session_state.ut_locked[ut] = True
    logger.info(
        "UT %s locked: %d confirmed rows, %d skipped",
        ut, len(new_confirmed), len(new_skipped),
    )


# ---------------------------------------------------------------------------
# Org hierarchy helpers
# ---------------------------------------------------------------------------

def build_org_map(orgs: list[dict]) -> dict[str, str]:
    """Build label→OrganizationID map from OrgHierarchy rows.
    Label format: "[OrgID] OrgName"
    """
    return {
        f"[{o['OrganizationID']}] {o['OrganizationName']}": o["OrganizationID"]
        for o in orgs
        if o.get("OrganizationID") and o.get("OrganizationName")
    }


def org_label(oid: str, org_map: dict) -> str:
    """Reverse-lookup: OrganizationID → display label."""
    for lbl, v in org_map.items():
        if v == oid:
            return lbl
    return oid  # fall back to raw ID if not in map


def org_multiselect(
    label: str,
    widget_key: str,
    current_org_ids: list[str],
    org_map: dict,
    disabled: bool = False,
) -> list[str]:
    """Render an org multiselect pre-filled from current_org_ids.
    Returns list of selected OrganizationIDs.
    """
    if org_map:
        # Convert IDs → labels for default
        defaults = [org_label(o, org_map) for o in current_org_ids if o]
        defaults = [d for d in defaults if d in org_map]
        sel = st.multiselect(
            label,
            list(org_map.keys()),
            default=defaults if widget_key not in st.session_state else None,
            key=widget_key,
            disabled=disabled,
        )
        return [org_map[s] for s in sel] if sel else [""]
    else:
        # No hierarchy loaded — fall back to free text
        raw = st.text_input(
            label,
            value=", ".join(current_org_ids) if current_org_ids else ORG_ID,
            key=widget_key,
            disabled=disabled,
            help="Upload OrganizationHierarchy.csv in tab 1 for a dropdown",
        )
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        return ids if ids else [""]


# ---------------------------------------------------------------------------
# Tab 1: Load Data
# ---------------------------------------------------------------------------

def tab_load():
    st.header("📂 Load Data")

    st.subheader("Researcher Roster")
    res_file = st.file_uploader(
        "ResearcherAndDocument.csv (required)", type=["csv"], key="res_csv_upload"
    )
    org_file = st.file_uploader(
        "OrganizationHierarchy.csv (optional)", type=["csv"], key="org_csv_upload"
    )

    st.subheader("WoS Data Source")
    source_choice = st.radio(
        "Input method", ["🌐 API (date range)", "📄 CSV upload"], horizontal=True
    )

    if source_choice == "🌐 API (date range)":
        wos_records, ready = _api_input_section()
    else:
        wos_records, ready = _csv_input_section()

    st.divider()
    col_load, col_reset = st.columns([2, 1])

    with col_reset:
        if st.button("🔄 Reset all data", use_container_width=True):
            _reset_session_data()
            st.rerun()

    with col_load:
        load_clicked = st.button(
            "⚙️ Process records",
            use_container_width=True,
            disabled=not (res_file and ready),
        )

    if load_clicked and res_file:
        with st.spinner("Building person index…"):
            try:
                person_list, max_pid, existing_pairs = build_person_index(res_file.read())
            except ValueError as exc:
                st.error(str(exc))
                return

        st.session_state.person_list    = person_list
        st.session_state.max_pid        = max_pid
        st.session_state.existing_pairs = existing_pairs
        st.session_state.staging_pid_counter = None

        if org_file:
            orgs = parse_org_hierarchy(org_file.read())
            st.session_state.orgs   = orgs
            st.session_state.org_map = build_org_map(orgs)
        else:
            st.session_state.orgs   = []
            st.session_state.org_map = {}

        with st.spinner("Extracting MUV author pairs…"):
            pairs = extract_muv_author_pairs(wos_records, cfg)

        if not pairs:
            st.warning("No MUV-affiliated authors found in the loaded records.")
            return

        with st.spinner("Matching authors to roster…"):
            batch = batch_process(pairs, person_list, existing_pairs, cfg, max_pid)

        st.session_state.all_pairs    = pairs
        st.session_state.batch_result = batch
        st.session_state.ut_list      = list(group_by_ut(pairs).keys())
        st.session_state.source       = source_choice

        _prepopulate_confirmed(batch["confirmed"])

        st.success(
            f"✅ Loaded {len(st.session_state.ut_list)} UTs — "
            f"{len(batch['confirmed'])} auto-confirmed, "
            f"{len(batch['needs_review'])} need review, "
            f"{len(batch['already_uploaded'])} already in MyOrg."
        )


def _api_input_section() -> tuple[list[dict], bool]:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key = st.text_input(
        "WoS API Key",
        type="password",
        placeholder="Enter your Clarivate API key",
        key="api_key_input",
    )
    if api_key:
        st.session_state["_api_key"] = api_key

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", value=date.today())

    extra_query = st.text_input(
        "Additional query terms (optional)", placeholder='e.g. SO="Journal Name"'
    )

    if api_key and start_date <= end_date:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔌 Test API key"):
                st.success("Valid." if validate_api_key(api_key, cfg) else "❌ Rejected.")
        with c2:
            if st.button("⬇️ Fetch from API", use_container_width=True):
                with st.spinner("Fetching from WoS API…"):
                    try:
                        client = WoSClient(api_key, cfg)
                        raw_records = list(client.query_date_range(
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            extra_query=extra_query,
                        ))
                        records = [r for r in (parse_wos_json_record(rr) for rr in raw_records) if r]
                        st.session_state["_api_records"] = records
                        st.success(f"Fetched {len(records)} records from API.")
                    except WoSAPIError as exc:
                        st.error(f"API error: {exc}")

    records = st.session_state.get("_api_records", [])
    ready   = bool(records) and bool(st.session_state.get("_api_key"))
    return records, ready


def _csv_input_section() -> tuple[list[dict], bool]:
    wos_file = st.file_uploader(
        "WoS export CSV (tab-delimited)", type=["csv", "txt"], key="wos_csv_upload"
    )
    records: list[dict] = []
    if wos_file:
        with st.spinner("Parsing WoS CSV…"):
            records = parse_wos_csv(wos_file.read())
        st.info(f"Parsed {len(records)} records from CSV.")
    return records, bool(records)


def _prepopulate_confirmed(confirmed_pairs: list[dict]):
    """Store auto-confirmed exact matches (org IDs from roster) into author_decs."""
    for pair in confirmed_pairs:
        key = _dec_key(pair)
        if key not in st.session_state.author_decs:
            st.session_state.author_decs[key] = {
                "action":  "confirm",
                "pid":     pair["suggested_pid"],
                "first":   pair["suggested_first"],
                "last":    pair["suggested_last"],
                "org_ids": pair.get("suggested_org_ids") or [ORG_ID],
                "doc_id":  pair["doc_id"],
                "is_new":  False,
                "note":    "AU exact match",
            }


# ---------------------------------------------------------------------------
# Tab 2: Review
# ---------------------------------------------------------------------------

def tab_review():
    st.header("🔍 Review")

    if not st.session_state.ut_list:
        st.info("Load data in tab 1 first.")
        return

    ut_list     = st.session_state.ut_list
    pairs_by_ut = group_by_ut(st.session_state.all_pairs)
    batch       = st.session_state.batch_result

    locked_count = sum(1 for ut in ut_list if st.session_state.ut_locked.get(ut))
    st.progress(locked_count / len(ut_list), text=f"{locked_count} / {len(ut_list)} UTs confirmed")

    ut_options  = [f"{'✅' if st.session_state.ut_locked.get(ut) else '🔲'} {ut}" for ut in ut_list]
    selected_idx = st.selectbox("Select UT", range(len(ut_list)), format_func=lambda i: ut_options[i])
    ut          = ut_list[selected_idx]
    pairs       = pairs_by_ut.get(ut, [])
    locked_flag = st.session_state.ut_locked.get(ut, False)

    st.markdown(f"**UT:** `{ut}` — {len(pairs)} MUV author(s)")

    # Already-in-MO summary
    already = [r for r in batch.get("already_uploaded", []) if r.get("doc_id") == ut]
    if already:
        with st.expander(f"⏭ {len(already)} already in MyOrg", expanded=False):
            for r in already:
                st.markdown(
                    f"`{r.get('raw_last','')}, {r.get('raw_first','')}` "
                    f"→ PID {r.get('suggested_pid','')} "
                    f"({r.get('suggested_org_ids', [''])[0] if r.get('suggested_org_ids') else ''})"
                )

    # Auto-confirmed summary
    auto = [r for r in batch.get("confirmed", []) if r.get("doc_id") == ut]
    if auto:
        with st.expander(f"✅ {len(auto)} auto-confirmed (exact match)", expanded=False):
            for r in auto:
                orgs = r.get("suggested_org_ids") or [ORG_ID]
                st.markdown(
                    f"`EXACT` **{r.get('suggested_last','')}, {r.get('suggested_first','')}** "
                    f"PID {r.get('suggested_pid','')} | Org: {', '.join(orgs)}"
                )

    if locked_flag:
        st.success("✅ This UT has been confirmed.")
        if st.button("↩️ Unlock to re-review", use_container_width=True):
            st.session_state.ut_locked[ut] = False
            st.session_state.confirmed_rows = [
                r for r in st.session_state.confirmed_rows if r.get("DocumentID") != ut
            ]
            st.session_state.skipped_rows = [
                r for r in st.session_state.skipped_rows if r.get("doc_id") != ut
            ]
            st.rerun()
        return

    # Author decision cards (needs_review only)
    rev_pairs = [p for p in pairs if _find_match_info(p, batch) and
                 _find_match_info(p, batch).get("match_type") in ("initial_expansion", "fuzzy", "new")]

    decisions: dict[str, dict] = {}

    for pair in rev_pairs:
        key        = _dec_key(pair)
        match_info = _find_match_info(pair, batch) or {}
        mt         = match_info.get("match_type", "new")
        existing   = st.session_state.author_decs.get(key)

        badge = {"initial_expansion": "🔵 INITIAL", "fuzzy": "🟡 FUZZY", "new": "🟢 NEW"}.get(mt, mt.upper())

        with st.expander(f"{badge}  {pair['display']}", expanded=True):
            _render_author_card(pair, key, match_info, existing, locked_flag, decisions)

    # Carry forward auto-confirmed decisions
    for pair in pairs:
        key = _dec_key(pair)
        if key in st.session_state.author_decs and key not in decisions:
            decisions[key] = st.session_state.author_decs[key]

    col_confirm, col_skip_all = st.columns([3, 1])
    with col_confirm:
        if st.button(f"✅ Confirm UT {ut}", type="primary", use_container_width=True):
            for key, dec in decisions.items():
                st.session_state.author_decs[key] = dec
            _store_confirmed_ut(ut, decisions)
            st.rerun()
    with col_skip_all:
        if st.button("⏭ Skip entire UT", use_container_width=True):
            skip_decs = {
                _dec_key(p): {"action": "skip", "display": p["display"],
                               "doc_id": p["doc_id"], "note": "UT skipped"}
                for p in pairs
            }
            _store_confirmed_ut(ut, skip_decs)
            st.rerun()


def _render_author_card(
    pair: dict,
    key: str,
    match_info: dict,
    existing: dict | None,
    locked: bool,
    decisions: dict,
):
    """Render decision widgets for one author that needs review."""
    mt             = match_info.get("match_type", "new")
    suggested_pid  = match_info.get("suggested_pid")
    suggested_first = match_info.get("suggested_first", pair["first"].title())
    suggested_last  = match_info.get("suggested_last",  pair["last"].title())
    suggested_orgs  = match_info.get("suggested_org_ids") or []
    score          = match_info.get("score", 0.0)
    affil_display  = pair.get("affil_display") or match_info.get("affil_display", "")

    # Show API affiliation for context
    if affil_display:
        st.caption(f"📍 API affiliation: {affil_display}")
    if pair.get("orcid"):
        st.caption(f"ORCID: {pair['orcid']}")

    action = st.radio(
        "Action",
        ["confirm", "skip"],
        index=0,
        key=f"action_{key}",
        horizontal=True,
        disabled=locked,
    )

    if action == "skip":
        decisions[key] = {
            "action":  "skip",
            "display": pair["display"],
            "doc_id":  pair["doc_id"],
            "note":    "User skipped",
        }
        return

    # ── Confirm path ─────────────────────────────────────────────────────────
    is_new = (mt == "new") or (suggested_pid is None)

    left, right = st.columns([3, 2])

    with left:
        if is_new:
            st.caption(f"🆕 New author — staging PID will be assigned on confirm")
            first_val = st.text_input("First name", value=suggested_first, key=f"first_{key}", disabled=locked)
            last_val  = st.text_input("Last name",  value=suggested_last,  key=f"last_{key}",  disabled=locked)
            pid_val   = None  # assigned at lock time
        else:
            # Candidate from match + override search
            st.caption(f"Score: {score:.2f} · suggested: {suggested_last}, {suggested_first} (PID {suggested_pid})")

            search_term = st.text_input(
                "🔍 Override — search roster",
                value="",
                key=f"search_{key}",
                placeholder="Type name to search…",
                disabled=locked,
            )
            person_list = st.session_state.person_list
            if search_term:
                term_norm = normalize_name(search_term)
                filtered  = [
                    p for p in person_list
                    if term_norm in normalize_name(p.get("LastName", ""))
                    or term_norm in normalize_name(p.get("FirstName", ""))
                ]
            else:
                filtered = [p for p in person_list if p["PersonID"] == suggested_pid]

            options = {
                f"{p['LastName']}, {p['FirstName']} (PID {p['PersonID']})": p
                for p in filtered[:20]
            }
            options["➕ Create as NEW PERSON"] = None

            pick = st.selectbox(
                "Select person",
                list(options.keys()),
                key=f"pick_{key}",
                disabled=locked,
            )

            chosen = options[pick]
            if chosen is None:
                # User chose to create new
                is_new     = True
                pid_val    = None
                first_val  = suggested_first
                last_val   = suggested_last
                suggested_orgs = []
                st.caption("🆕 Will be created as new author")
            else:
                pid_val       = chosen["PersonID"]
                first_val     = chosen["FirstName"]
                last_val      = chosen["LastName"]
                suggested_orgs = chosen.get("OrganizationIDs") or (
                    [chosen["OrganizationID"]] if chosen.get("OrganizationID") else []
                )

    with right:
        org_map = st.session_state.get("org_map", {})
        if is_new:
            # New author: choose org from hierarchy (or free text fallback)
            st.markdown("**OrganizationID** for new author")
            if affil_display:
                st.caption(f"Hint from API: {affil_display}")
            org_ids_final = org_multiselect(
                "Organisation(s)",
                f"org_new_{key}",
                [ORG_ID],
                org_map,
                disabled=locked,
            )
        else:
            # Existing author: pre-fill from roster, allow correction via hierarchy
            st.markdown("**OrganizationID(s)**")
            org_ids_final = org_multiselect(
                "Organisation(s)",
                f"org_ex_{key}",
                suggested_orgs or [ORG_ID],
                org_map,
                disabled=locked,
            )

    decisions[key] = {
        "action":  "confirm",
        "pid":     pid_val,
        "first":   first_val,
        "last":    last_val,
        "org_ids": org_ids_final,
        "doc_id":  pair["doc_id"],
        "is_new":  is_new,
        "note":    f"{mt} match" if not is_new else "New author",
    }


def _find_match_info(pair: dict, batch: dict) -> dict | None:
    if batch is None:
        return None
    for pool in ("confirmed", "needs_review", "already_uploaded"):
        for r in batch.get(pool, []):
            if (r.get("doc_id") == pair["doc_id"]
                    and r.get("raw_last")  == pair["last"]
                    and r.get("raw_first") == pair["first"]):
                return r
    return None


# ---------------------------------------------------------------------------
# Tab 3: Export
# ---------------------------------------------------------------------------

def tab_export():
    st.header("⬇️ Export")

    if not st.session_state.confirmed_rows and not st.session_state.skipped_rows:
        st.info("No confirmed UTs yet. Review and confirm UTs in tab 2.")
        return

    upload_rows = _resolve_staging_pids(st.session_state.confirmed_rows)
    upload_rows = deduplicate_upload_rows(upload_rows)

    confirmed_count = len([r for r in upload_rows])
    skipped_count   = len(st.session_state.skipped_rows)
    new_count       = len([r for r in upload_rows if r.get("is_new")])

    col1, col2, col3 = st.columns(3)
    col1.metric("4UP rows", confirmed_count)
    col2.metric("2SKIP rows", skipped_count)
    col3.metric("New authors", new_count)

    upload_cols = ["PersonID", "FirstName", "LastName", "OrganizationID", "DocumentID"]
    upload_df   = pd.DataFrame([{c: r.get(c, "") for c in upload_cols} for r in upload_rows])

    full_rows = upload_rows + st.session_state.skipped_rows
    full_df   = pd.DataFrame(full_rows)

    st.subheader("upload.csv preview")
    st.dataframe(upload_df.head(100), use_container_width=True)

    # Warn about rows with empty OrganizationID
    empty_org = upload_df[upload_df["OrganizationID"].astype(str).str.strip() == ""]
    if not empty_org.empty:
        st.warning(f"⚠️ {len(empty_org)} row(s) have an empty OrganizationID — review new authors in tab 2.")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download upload.csv",
            data=upload_df.to_csv(index=False).encode("utf-8"),
            file_name="upload.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇️ Download full_output.csv",
            data=full_df.to_csv(index=False).encode("utf-8"),
            file_name="full_output.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.batch_result and st.session_state.batch_result.get("already_uploaded"):
        au = st.session_state.batch_result["already_uploaded"]
        with st.expander(f"ℹ️ Already in MyOrg ({len(au)} pairs)"):
            st.dataframe(pd.DataFrame(au), use_container_width=True)


def _resolve_staging_pids(rows: list[dict]) -> list[dict]:
    """Assign real staging PIDs to new authors (pid=None rows).
    Same new person across multiple UTs gets the same PID.
    """
    resolved: list[dict] = []
    new_person_map: dict[tuple, int] = {}

    for row in rows:
        if row.get("PersonID") is None and row.get("is_new"):
            key = (
                normalize_name(row.get("FirstName", "")),
                normalize_name(row.get("LastName", "")),
            )
            if key not in new_person_map:
                new_person_map[key] = _next_staging_pid()
            row = {**row, "PersonID": new_person_map[key]}
        resolved.append(row)
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="WoS → MyOrg", page_icon="🔬", layout="wide")
st.title("🔬 WoS → InCites My Organization (v4)")

tab1, tab2, tab3 = st.tabs(["📂 1 · Load Data", "🔍 2 · Review", "⬇️ 3 · Export"])

with tab1:
    tab_load()

with tab2:
    tab_review()

with tab3:
    tab_export()
