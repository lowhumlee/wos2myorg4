"""
app.py — Streamlit UI for WoS → MyOrg pipeline (v4).

Three tabs:
  1. Load Data  — API date-range query OR CSV upload + ResearcherAndDocument.csv
  2. Review     — UT-centric author decisions
  3. Export     — download upload.csv + full output
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
cfg = load_config(CFG_PATH)
ORG_ID = cfg.get("organization_id", "MED_UNIV_VARNA")

# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
KEYS_DEFAULTS = {
    "confirmed_rows": [],        # 4UP rows accumulated across locked UTs
    "skipped_rows": [],          # 2SKIP rows
    "ut_locked": {},             # ut → True when confirmed
    "author_decs": {},           # (norm_last+norm_first, ut) → decision dict
    "existing_pairs": set(),
    "max_pid": 0,
    "staging_pid_counter": None, # int once first new author assigned
    "person_list": [],
    "all_pairs": [],             # all MUV author-document pairs
    "batch_result": None,        # output of batch_process
    "ut_list": [],               # ordered list of UTs
    "source": None,              # "api" or "csv"
}

for k, v in KEYS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_staging_pid() -> int:
    """Return the next available staging PersonID (above max_pid)."""
    if not isinstance(st.session_state.staging_pid_counter, int):
        st.session_state.staging_pid_counter = st.session_state.max_pid + 1
    else:
        st.session_state.staging_pid_counter += 1
    return st.session_state.staging_pid_counter


def _dec_key(pair: dict) -> str:
    # pair dicts use last/first; match result dicts use raw_last/raw_first
    last = pair.get("last") or pair.get("raw_last", "")
    first = pair.get("first") or pair.get("raw_first", "")
    return f"{last}|{first}|{pair['doc_id']}"


def _reset_session_data():
    """Clear all loaded data to allow a fresh load."""
    for k, v in KEYS_DEFAULTS.items():
        st.session_state[k] = v() if callable(v) else (
            v.copy() if isinstance(v, (list, dict, set)) else v
        )
    st.session_state.pop("_api_records", None)
    st.session_state.pop("_api_key", None)


def _store_confirmed_ut(ut: str, decisions: dict[str, dict]):
    """Lock a UT and accumulate its rows into session state."""
    if st.session_state.ut_locked.get(ut):
        return

    org_id = ORG_ID
    ut_confirmed: list[dict] = []
    ut_skipped: list[dict] = []

    for key, dec in decisions.items():
        action = dec.get("action")
        if action == "skip":
            ut_skipped.append({**dec, "Status": "2SKIP", "Note": dec.get("note", "")})
            continue

        pid = dec.get("pid")
        first = dec.get("first", "")
        last = dec.get("last", "")
        doc_id = dec.get("doc_id", ut)

        row = build_upload_row(pid, first, last, doc_id, org_id)
        row["Status"] = "4UP"
        row["Note"] = dec.get("note", "")
        row["is_new"] = dec.get("is_new", False)
        ut_confirmed.append(row)

    st.session_state.confirmed_rows.extend(ut_confirmed)
    st.session_state.skipped_rows.extend(ut_skipped)
    st.session_state.ut_locked[ut] = True
    logger.info("UT %s locked: %d confirmed, %d skipped", ut, len(ut_confirmed), len(ut_skipped))


# ---------------------------------------------------------------------------
# Tab 1: Load Data
# ---------------------------------------------------------------------------

def tab_load():
    st.header("📂 Load Data")

    # --- ResearcherAndDocument.csv (always required) ---
    st.subheader("Researcher Roster")
    res_file = st.file_uploader(
        "ResearcherAndDocument.csv (required)",
        type=["csv"],
        key="res_csv_upload",
    )
    org_file = st.file_uploader(
        "OrganizationHierarchy.csv (optional)",
        type=["csv"],
        key="org_csv_upload",
    )

    # --- Source selector ---
    st.subheader("WoS Data Source")
    source_choice = st.radio(
        "How do you want to load WoS records?",
        ["🌐 API (date range)", "📄 CSV upload"],
        horizontal=True,
    )

    wos_records: list[dict] = []
    ready = False

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

        st.session_state.person_list = person_list
        st.session_state.max_pid = max_pid
        st.session_state.existing_pairs = existing_pairs
        st.session_state.staging_pid_counter = None

        if org_file:
            _ = parse_org_hierarchy(org_file.read())  # available for future use

        with st.spinner("Extracting MUV author pairs…"):
            pairs = extract_muv_author_pairs(wos_records, cfg)

        if not pairs:
            st.warning("No MUV-affiliated authors found in the loaded records.")
            return

        with st.spinner("Matching authors to roster…"):
            batch = batch_process(pairs, person_list, existing_pairs, cfg, max_pid)

        st.session_state.all_pairs = pairs
        st.session_state.batch_result = batch
        st.session_state.ut_list = list(group_by_ut(pairs).keys())
        st.session_state.source = source_choice

        # Pre-populate decisions for auto-confirmed pairs
        _prepopulate_confirmed(batch["confirmed"])

        st.success(
            f"✅ Loaded {len(st.session_state.ut_list)} UTs — "
            f"{len(batch['confirmed'])} auto-confirmed, "
            f"{len(batch['needs_review'])} need review, "
            f"{len(batch['already_uploaded'])} already in MyOrg."
        )


def _api_input_section() -> tuple[list[dict], bool]:
    """Render API input widgets and return (records, ready)."""
    api_key = st.text_input(
        "WoS API Key",
        type="password",
        placeholder="Enter your Clarivate API key",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", value=date.today())

    extra_query = st.text_input(
        "Additional query terms (optional)",
        placeholder='e.g. SO="Journal Name"',
    )

    # Persist api_key across rerenders
    if api_key:
        st.session_state["_api_key"] = api_key

    records: list[dict] = []

    if api_key and start_date <= end_date:
        if st.button("🔌 Test API key"):
            if validate_api_key(api_key, cfg):
                st.success("API key valid.")
            else:
                st.error("API key rejected or unreachable.")

        if st.button("⬇️ Fetch from API", use_container_width=False):
            with st.spinner("Fetching from WoS API…"):
                try:
                    client = WoSClient(api_key, cfg)
                    raw_records = list(
                        client.query_date_range(
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            extra_query=extra_query,
                        )
                    )
                    records = [
                        r for r in (parse_wos_json_record(rr) for rr in raw_records) if r
                    ]
                    st.session_state["_api_records"] = records
                    st.success(f"Fetched {len(records)} records from API.")
                except WoSAPIError as exc:
                    st.error(f"API error: {exc}")

        if "_api_records" in st.session_state:
            records = st.session_state["_api_records"]

    ready = bool(st.session_state.get("_api_records")) and bool(st.session_state.get("_api_key"))
    return records, ready


def _csv_input_section() -> tuple[list[dict], bool]:
    """Render CSV upload widget and return (records, ready)."""
    wos_file = st.file_uploader(
        "WoS export CSV (tab-delimited)",
        type=["csv", "txt"],
        key="wos_csv_upload",
    )
    records: list[dict] = []
    if wos_file:
        with st.spinner("Parsing WoS CSV…"):
            records = parse_wos_csv(wos_file.read())
        st.info(f"Parsed {len(records)} records from CSV.")
    return records, bool(records)


def _prepopulate_confirmed(confirmed_pairs: list[dict]):
    """Store auto-confirmed exact matches into author_decs."""
    for pair in confirmed_pairs:
        key = _dec_key(pair)
        if key not in st.session_state.author_decs:
            st.session_state.author_decs[key] = {
                "action": "confirm",
                "pid": pair["suggested_pid"],
                "first": pair["suggested_first"],
                "last": pair["suggested_last"],
                "doc_id": pair["doc_id"],
                "is_new": False,
                "note": "AU exact match",
            }


# ---------------------------------------------------------------------------
# Tab 2: Review
# ---------------------------------------------------------------------------

def tab_review():
    st.header("🔍 Review")

    if not st.session_state.ut_list:
        st.info("Load data in tab 1 first.")
        return

    ut_list = st.session_state.ut_list
    pairs_by_ut = group_by_ut(st.session_state.all_pairs)
    batch = st.session_state.batch_result

    # Needs-review index by doc_id
    needs_review_index: dict[str, list[dict]] = {}
    for p in batch["needs_review"]:
        needs_review_index.setdefault(p["doc_id"], []).append(p)

    # Progress
    locked = sum(1 for ut in ut_list if st.session_state.ut_locked.get(ut))
    st.progress(locked / len(ut_list), text=f"{locked} / {len(ut_list)} UTs confirmed")

    # UT selector
    ut_options = [
        f"{'✅' if st.session_state.ut_locked.get(ut) else '🔲'} {ut}"
        for ut in ut_list
    ]
    selected_idx = st.selectbox(
        "Select UT to review",
        range(len(ut_list)),
        format_func=lambda i: ut_options[i],
    )
    ut = ut_list[selected_idx]
    pairs = pairs_by_ut.get(ut, [])

    st.markdown(f"**UT:** `{ut}` — {len(pairs)} MUV author(s)")
    locked_flag = st.session_state.ut_locked.get(ut, False)
    if locked_flag:
        st.success("✅ This UT has been confirmed.")

    decisions: dict[str, dict] = {}

    for pair in pairs:
        key = _dec_key(pair)
        existing_dec = st.session_state.author_decs.get(key)

        with st.expander(
            f"{'🔒 ' if locked_flag else ''}{pair['display']} — {pair['source']}",
            expanded=not locked_flag,
        ):
            _render_author_row(pair, key, existing_dec, locked_flag, decisions)

    if not locked_flag:
        if st.button(f"✅ Confirm UT {ut}", type="primary", use_container_width=True):
            # Save all decisions first
            for key, dec in decisions.items():
                st.session_state.author_decs[key] = dec
            _store_confirmed_ut(ut, decisions)
            st.rerun()
    else:
        if st.button("↩️ Unlock this UT (re-review)", use_container_width=True):
            st.session_state.ut_locked[ut] = False
            # Remove rows for this UT from confirmed/skipped
            st.session_state.confirmed_rows = [
                r for r in st.session_state.confirmed_rows if r.get("DocumentID") != ut
            ]
            st.session_state.skipped_rows = [
                r for r in st.session_state.skipped_rows if r.get("doc_id") != ut
            ]
            st.rerun()


def _render_author_row(
    pair: dict,
    key: str,
    existing_dec: dict | None,
    locked: bool,
    decisions: dict,
):
    """Render decision widgets for one author-document pair."""
    # Find match info from batch
    batch = st.session_state.batch_result
    match_info = _find_match_info(pair, batch)
    mt = match_info.get("match_type", "new") if match_info else "new"
    suggested_pid = match_info.get("suggested_pid") if match_info else None
    suggested_first = match_info.get("suggested_first", pair["first"].title()) if match_info else pair["first"].title()
    suggested_last = match_info.get("suggested_last", pair["last"].title()) if match_info else pair["last"].title()
    score = match_info.get("score", 0.0) if match_info else 0.0

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"**WoS name:** {pair['display']}")
        st.markdown(f"**Match type:** `{mt}` (score: {score:.2f})")
        if pair.get("orcid"):
            st.markdown(f"**ORCID:** {pair['orcid']}")

    with col_b:
        if locked:
            dec = existing_dec or {}
            action = dec.get("action", "skip")
            st.markdown(
                f"Decision: **{'✅ Confirm' if action == 'confirm' else '⏭️ Skip'}** "
                f"— {dec.get('first', '')} {dec.get('last', '')} (PID {dec.get('pid', '—')})"
            )
            decisions[key] = dec
            return

        # Action radio
        action = st.radio(
            "Action",
            ["confirm", "skip"],
            index=0 if mt != "new" else 0,
            key=f"action_{key}",
            horizontal=True,
            disabled=locked,
        )

        if action == "confirm":
            # Person identity
            is_new = (mt == "new") or (suggested_pid is None)

            if is_new:
                st.caption("🆕 New author — a staging PID will be assigned on confirm.")
                first_val = st.text_input("First name", value=suggested_first, key=f"first_{key}", disabled=locked)
                last_val = st.text_input("Last name", value=suggested_last, key=f"last_{key}", disabled=locked)
                pid_val = None  # assigned at lock time
            else:
                # Search / override from roster
                search_term = st.text_input(
                    "Search roster (override match)",
                    value="",
                    key=f"search_{key}",
                    disabled=locked,
                    placeholder="Type to search…",
                )

                person_list = st.session_state.person_list
                if search_term:
                    term_norm = normalize_name(search_term)
                    filtered = [
                        p for p in person_list
                        if term_norm in normalize_name(p.get("LastName", ""))
                        or term_norm in normalize_name(p.get("FirstName", ""))
                    ]
                else:
                    # Show suggested match at top
                    filtered = [
                        p for p in person_list if p["PersonID"] == suggested_pid
                    ]

                options = {
                    f"{p['LastName']}, {p['FirstName']} (PID {p['PersonID']})": p
                    for p in filtered[:20]
                }
                if not options:
                    st.caption("No matching persons found.")
                    first_val = suggested_first
                    last_val = suggested_last
                    pid_val = suggested_pid
                else:
                    pick_key = f"pick_{key}"
                    selection = st.selectbox(
                        "Select person",
                        list(options.keys()),
                        key=pick_key,
                        disabled=locked,
                    )
                    chosen = options[selection]
                    pid_val = chosen["PersonID"]
                    first_val = chosen["FirstName"]
                    last_val = chosen["LastName"]

            decisions[key] = {
                "action": "confirm",
                "pid": pid_val,
                "first": first_val,
                "last": last_val,
                "doc_id": pair["doc_id"],
                "is_new": is_new,
                "note": f"{mt} match" if not is_new else "New author",
            }

        else:  # skip
            decisions[key] = {
                "action": "skip",
                "doc_id": pair["doc_id"],
                "note": "User skipped",
                "display": pair["display"],
            }


def _find_match_info(pair: dict, batch: dict) -> dict | None:
    """Locate the batch result entry for a given pair."""
    if batch is None:
        return None
    all_results = (
        batch.get("confirmed", [])
        + batch.get("needs_review", [])
        + batch.get("already_uploaded", [])
    )
    for r in all_results:
        if r.get("doc_id") == pair["doc_id"] and r.get("raw_last") == pair["last"] and r.get("raw_first") == pair["first"]:
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

    # Resolve staging PIDs for new authors
    upload_rows = _resolve_staging_pids(st.session_state.confirmed_rows)
    upload_rows = deduplicate_upload_rows(upload_rows)

    # Summary
    confirmed_count = len([r for r in upload_rows if r.get("Status") == "4UP"])
    skipped_count = len(st.session_state.skipped_rows)
    new_count = len([r for r in upload_rows if r.get("is_new")])

    col1, col2, col3 = st.columns(3)
    col1.metric("4UP rows", confirmed_count)
    col2.metric("2SKIP rows", skipped_count)
    col3.metric("New authors", new_count)

    # Upload CSV (5 columns only)
    upload_cols = ["PersonID", "FirstName", "LastName", "OrganizationID", "DocumentID"]
    upload_df = pd.DataFrame(
        [{c: r.get(c, "") for c in upload_cols} for r in upload_rows]
    )

    # Full output (with Status + Note)
    full_rows = upload_rows + st.session_state.skipped_rows
    full_df = pd.DataFrame(full_rows)

    st.subheader("upload.csv preview")
    st.dataframe(upload_df.head(50), use_container_width=True)

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

    # Already uploaded summary
    if st.session_state.batch_result and st.session_state.batch_result.get("already_uploaded"):
        with st.expander(f"ℹ️ Already in MyOrg ({len(st.session_state.batch_result['already_uploaded'])} pairs)"):
            st.dataframe(
                pd.DataFrame(st.session_state.batch_result["already_uploaded"]),
                use_container_width=True,
            )


def _resolve_staging_pids(rows: list[dict]) -> list[dict]:
    """Assign real staging PIDs to new authors (pid=None rows).

    Groups by (first, last) so the same new person across multiple UTs
    gets a consistent PID.
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

st.set_page_config(
    page_title="WoS → MyOrg",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 WoS → InCites My Organization (v4)")

tab1, tab2, tab3 = st.tabs(["📂 1 · Load Data", "🔍 2 · Review", "⬇️ 3 · Export"])

with tab1:
    tab_load()

with tab2:
    tab_review()

with tab3:
    tab_export()
