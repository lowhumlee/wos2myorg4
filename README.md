# WoS → InCites My Organization (v4)

Streamlit app for Medical University of Varna (MUV) to ingest Web of Science data into Clarivate InCites My Organization.

## Features

- **Dual input**: WoS Expanded API (date-range query) or WoS CSV export upload
- **UT-centric review**: Confirm or skip authors publication by publication
- **Author matching**: Exact → initial expansion → fuzzy → new author
- **Safe PID assignment**: Staging IDs for new authors above `max(PersonID)`
- **Clean export**: `upload.csv` (5 columns) + `full_output.csv` with status notes

## File Structure

```
wos2myorg4/
├── app.py            # Streamlit UI
├── api.py            # WoS Expanded API client
├── core.py           # Parsing + MUV extraction + batch processing
├── matching.py       # All name matching logic
├── config.json       # Affiliation patterns + thresholds
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input Files

| File | Required | Source |
|------|----------|--------|
| `ResearcherAndDocument.csv` | ✅ | Export from InCites My Organization |
| `OrganizationHierarchy.csv` | ❌ | Export from InCites My Organization |
| WoS CSV export | ✅ (if not using API) | WoS Full Record export (tab-delimited) |

## Workflow

```
Tab 1: Load Data
  ├── Upload ResearcherAndDocument.csv
  ├── Choose source: API (date range) OR CSV upload
  └── Click "Process records"

Tab 2: Review
  ├── Work through UTs one by one
  ├── Confirm or skip each MUV author
  └── Click "✅ Confirm UT" to lock

Tab 3: Export
  ├── Download upload.csv  → import into InCites MyOrg
  └── Download full_output.csv → audit trail
```

## Output Format

**upload.csv** (strict MyOrg format):
```
PersonID,FirstName,LastName,OrganizationID,DocumentID
2131,Silvia,Gancheva,MED_UNIV_VARNA,WOS:001234567890000
```

## Match Types

| Type | Meaning | Auto-confirmed? |
|------|---------|-----------------|
| `exact` | Last + first normalised match | ✅ Yes |
| `initial_expansion` | WoS initials expand to master full name | 🔍 Review |
| `fuzzy` | High surname + first similarity | 🔍 Review |
| `new` | Not found in roster | 🔍 Review + new PID |
| `probable_duplicate` | Already in MyOrg | ⏭️ Skipped |

## Config

Edit `config.json` to adjust:
- `affiliation_patterns` — MUV name variants
- `hospital_partners` — associated hospitals (Tier 2 C3 fallback)
- `matching.fuzzy_threshold` — (default 0.85)
- `matching.initial_expansion_threshold` — (default 0.80)
- `api.base_url`, `api.page_size`, `api.max_retries`

## Deployment (Streamlit Cloud)

1. Push to GitHub (private repo)
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Add `CLARIVATE_API_KEY` to Streamlit secrets if desired (app also accepts it via UI)
