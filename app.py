import streamlit as st
import pandas as pd
from datetime import date, datetime, timezone
import itertools
import random
import re
from typing import Dict, List, Tuple, Optional
from streamlit_gsheets import GSheetsConnection

# =========================
# Config
# =========================
st.set_page_config(page_title="Volunteer Scheduler V1", layout="wide")
st.title("Volunteer Scheduler V1 (Blackjack Fairness)")

conn = st.connection("gsheets", type=GSheetsConnection)

DEFAULT_ROSTER_WS = "Roster"
DEFAULT_LOG_WS = "Schedule_Log"

ROSTER_COLS = [
    "Name",
    "RoleCapability",
    "TrueCount",
    "Email",
    "SkillLevel",
    "MedicalRestrictions",
    "PreferredCameras",
    "AvoidCameras",
    "LastRun",
]

LOG_COLS = [
    "Date",
    "Name",
    "RoleAssigned",
    "GeneratedAtUTC",
    "Notes",
]

CAMERAS_ALL = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6"]


# =========================
# Helpers
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_list_cell(x) -> List[str]:
    """Comma-separated string -> list of normalized tokens."""
    if pd.isna(x) or x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def normalize_camera_token(x: str) -> str:
    """Normalize user camera strings like '1', 'cam1', 'Cam 1' -> 'Cam1'."""
    if not x:
        return ""
    s = str(x).strip().lower().replace(" ", "")
    m = re.match(r"^(cam)?([1-6])$", s)
    if m:
        return f"Cam{m.group(2)}"
    return str(x).strip()


def parse_cameras_cell(x) -> List[str]:
    cams = []
    for token in normalize_list_cell(x):
        cam = normalize_camera_token(token)
        if cam in CAMERAS_ALL:
            cams.append(cam)
    return cams


def require_columns(df: pd.DataFrame, required: List[str], which: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{which} sheet missing columns: {missing}. Expected exactly: {required}")


def load_roster(roster_ws: str) -> pd.DataFrame:
    df = conn.read(worksheet=roster_ws)
    require_columns(df, ROSTER_COLS, "Roster")

    df = df[ROSTER_COLS].copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["TrueCount"] = pd.to_numeric(df["TrueCount"], errors="coerce").fillna(0).astype(int)
    df["PreferredCameras"] = df["PreferredCameras"].apply(parse_cameras_cell)
    df["AvoidCameras"] = df["AvoidCameras"].apply(parse_cameras_cell)
    df["LastRun"] = df["LastRun"].fillna("").astype(str).str.strip()
    return df


def load_log(log_ws: str) -> pd.DataFrame:
    try:
        df = conn.read(worksheet=log_ws)
    except Exception:
        df = pd.DataFrame(columns=LOG_COLS)

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=LOG_COLS)

    require_columns(df, LOG_COLS, "Schedule_Log")
    df = df[LOG_COLS].copy()
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    df["RoleAssigned"] = df["RoleAssigned"].fillna("").astype(str).str.strip()
    df["GeneratedAtUTC"] = df["GeneratedAtUTC"].fillna("").astype(str).str.strip()
    df["Notes"] = df["Notes"].fillna("").astype(str)
    return df


def has_capability(role_capability_cell: str, required_token: str) -> bool:
    tokens = [t.strip().lower() for t in str(role_capability_cell).split(",") if t.strip()]
    return required_token.lower() in tokens


def build_coverage_state(
    log_df: pd.DataFrame,
    window_serving_days: int,
) -> Dict[str, set]:
    """
    Returns dict: name -> set(cameras_seen_in_window)
    Window is last N unique dates in the log.
    """
    if log_df is None or len(log_df) == 0:
        return {}

    dates = (
        log_df["Date"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Keep unique dates in order of appearance
    unique_dates = []
    seen = set()
    for d in dates:
        if d and d not in seen:
            unique_dates.append(d)
            seen.add(d)

    # Take last N serving days
    window_dates = set(unique_dates[-window_serving_days:]) if unique_dates else set()

    window_df = log_df[log_df["Date"].astype(str).isin(window_dates)].copy()

    cov: Dict[str, set] = {}
    for _, r in window_df.iterrows():
        n = str(r["Name"]).strip()
        cam = str(r["RoleAssigned"]).strip()
        if n and cam in CAMERAS_ALL:
            cov.setdefault(n, set()).add(cam)
    return cov


def parse_note_field(notes: str, key: str) -> Optional[str]:
    """
    Notes are stored like:
    "BatchID=...;Delta=-3;PrevLastRun=Cam2;..."
    """
    if not notes:
        return None
    parts = [p.strip() for p in notes.split(";") if p.strip()]
    for p in parts:
        if p.lower().startswith(key.lower() + "="):
            return p.split("=", 1)[1].strip()
    return None


def score_assignment(
    person_row: pd.Series,
    role: str,
    truecount: int,
    role_value: int,
    preferred_bonus: int,
    avoid_penalty: int,
    back_to_back_penalty: int,
    coverage_penalty: int,
    hard_avoid: bool,
    hard_back_to_back: bool,
    last_run_map: Dict[str, str],
    coverage_map: Dict[str, set],
) -> Tuple[bool, int, List[str]]:
    """
    Returns: (is_valid, score, reasons)
    Higher score is better.
    """
    name = str(person_row["Name"]).strip()
    reasons: List[str] = []

    score = 0

    # Due factor: higher TrueCount should favor premium roles (positive role_value),
    # and disfavor low roles (negative role_value).
    # This works because score += truecount * role_value.
    score += truecount * role_value

    prefs = person_row["PreferredCameras"]
    avoids = person_row["AvoidCameras"]

    if role in prefs:
        score += preferred_bonus
        reasons.append(f"preferred {role}")

    if role in avoids:
        if hard_avoid:
            return False, -10**9, [f"avoid {role} (blocked)"]
        score -= avoid_penalty
        reasons.append(f"avoid {role} (penalty)")

    last_run = last_run_map.get(name, "")
    if last_run == role:
        if hard_back_to_back:
            return False, -10**9, [f"back-to-back {role} (blocked)"]
        score -= back_to_back_penalty
        reasons.append(f"back-to-back {role} (penalty)")

    # Coverage: penalize assigning a camera that they've already seen in window
    seen = coverage_map.get(name, set())
    if role in seen:
        score -= coverage_penalty
        reasons.append(f"coverage repeat {role} (penalty)")
    else:
        reasons.append(f"coverage helps {role}")

    return True, score, reasons


def generate_options(
    team_df: pd.DataFrame,
    active_roles: List[str],
    role_values: Dict[str, int],
    preferred_bonus: int,
    avoid_penalty: int,
    back_to_back_penalty: int,
    coverage_penalty: int,
    hard_avoid: bool,
    hard_back_to_back: bool,
    enable_coverage: bool,
    coverage_map: Dict[str, set],
    locks: Dict[str, str],  # role -> name
    n_options: int,
    n_random_samples: int,
) -> List[Dict]:
    """
    Returns list of options:
    {
      "assignments": List[{"Name":..., "RoleAssigned":...}],
      "score": int,
      "why": List[str],
    }
    """
    team_names = team_df["Name"].tolist()
    if len(team_names) != len(active_roles):
        raise ValueError("Team size must equal number of active cameras.")

    # Build last run map from roster
    last_run_map = {str(r["Name"]).strip(): str(r["LastRun"]).strip() for _, r in team_df.iterrows()}

    # Validate locks
    for role, locked_name in locks.items():
        if role not in active_roles:
            raise ValueError(f"Locked role {role} is not active.")
        if locked_name not in team_names:
            raise ValueError(f"Locked name {locked_name} is not in the selected team.")

    # Build fixed assignments from locks
    fixed_role_to_name = dict(locks)
    fixed_name_to_role = {v: k for k, v in fixed_role_to_name.items()}

    remaining_roles = [r for r in active_roles if r not in fixed_role_to_name]
    remaining_names = [n for n in team_names if n not in fixed_name_to_role]

    # Pre-index rows
    row_by_name = {str(r["Name"]).strip(): r for _, r in team_df.iterrows()}

    def score_full_mapping(role_to_name: Dict[str, str]) -> Tuple[int, List[str], bool]:
        total = 0
        why_lines: List[str] = []
        valid = True

        # Coverage can be disabled
        cov_map = coverage_map if enable_coverage else {}

        for role, name in role_to_name.items():
            row = row_by_name[name]
            tc = int(row["TrueCount"])
            rv = int(role_values.get(role, 0))

            ok, sc, reasons = score_assignment(
                person_row=row,
                role=role,
                truecount=tc,
                role_value=rv,
                preferred_bonus=preferred_bonus,
                avoid_penalty=avoid_penalty,
                back_to_back_penalty=back_to_back_penalty,
                coverage_penalty=coverage_penalty,
                hard_avoid=hard_avoid,
                hard_back_to_back=hard_back_to_back,
                last_run_map=last_run_map,
                coverage_map=cov_map,
            )

            if not ok:
                valid = False
                break

            total += sc
            why_lines.append(f"{name} -> {role}: " + ", ".join(reasons))

        return total, why_lines, valid

    # Strategy:
    # If remaining permutation count is small, enumerate all.
    # Else sample random permutations.
    perm_count = 1
    for i in range(2, len(remaining_roles) + 1):
        perm_count *= i

    candidates = []
    seen_maps = set()

    def add_candidate(role_to_name: Dict[str, str]):
        # Make a stable key to avoid duplicates
        key = tuple(sorted(role_to_name.items()))
        if key in seen_maps:
            return
        seen_maps.add(key)

        score, why, valid = score_full_mapping(role_to_name)
        if valid:
            candidates.append({"role_to_name": role_to_name, "score": score, "why": why})

    if perm_count <= 2000:
        for perm in itertools.permutations(remaining_names):
            role_to_name = dict(fixed_role_to_name)
            for role, name in zip(remaining_roles, perm):
                role_to_name[role] = name
            add_candidate(role_to_name)
    else:
        for _ in range(max(n_random_samples, n_options * 50)):
            random.shuffle(remaining_names)
            role_to_name = dict(fixed_role_to_name)
            for role, name in zip(remaining_roles, remaining_names):
                role_to_name[role] = name
            add_candidate(role_to_name)

    if not candidates:
        raise ValueError("No valid schedules found under current rules. Try relaxing hard blocks or changing locks.")

    # Sort by score descending, take top N unique
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:n_options]

    options = []
    for c in top:
        # Create consistent display order by active_roles
        assignments = [{"Name": c["role_to_name"][role], "RoleAssigned": role} for role in active_roles]
        options.append({"assignments": assignments, "score": c["score"], "why": c["why"]})

    return options


def apply_commit(
    roster_df: pd.DataFrame,
    log_df: pd.DataFrame,
    roster_ws: str,
    log_ws: str,
    service_date: date,
    commit_option: Dict,
    role_truecount_delta: Dict[str, int],
    batch_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Writes back:
      - roster TrueCount and LastRun updated for assigned people
      - schedule log appended with rows containing BatchID and deltas inside Notes
    Returns (new_roster_df, new_log_df)
    """
    now_iso = utc_now_iso()
    date_str = service_date.isoformat()

    updated_roster = roster_df.copy()
    updated_log = log_df.copy()

    # For undo: record prev TrueCount and prev LastRun in Notes per row.
    for a in commit_option["assignments"]:
        name = a["Name"]
        role = a["RoleAssigned"]

        idx = updated_roster.index[updated_roster["Name"] == name]
        if len(idx) != 1:
            raise ValueError(f"Roster row not found or duplicated for name: {name}")
        i = idx[0]

        prev_tc = int(updated_roster.at[i, "TrueCount"])
        prev_lr = str(updated_roster.at[i, "LastRun"]).strip()
        delta = int(role_truecount_delta.get(role, 0))
        new_tc = prev_tc + delta

        updated_roster.at[i, "TrueCount"] = int(new_tc)
        updated_roster.at[i, "LastRun"] = role

        notes = f"BatchID={batch_id};Delta={delta};PrevTrueCount={prev_tc};PrevLastRun={prev_lr}"

        new_row = {
            "Date": date_str,
            "Name": name,
            "RoleAssigned": role,
            "GeneratedAtUTC": now_iso,
            "Notes": notes,
        }
        updated_log = pd.concat([updated_log, pd.DataFrame([new_row])], ignore_index=True)

    # Write back to sheets
    conn.update(worksheet=roster_ws, data=updated_roster[ROSTER_COLS])
    conn.update(worksheet=log_ws, data=updated_log[LOG_COLS])

    return updated_roster, updated_log


def undo_last_commit(
    roster_df: pd.DataFrame,
    log_df: pd.DataFrame,
    roster_ws: str,
    log_ws: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finds the most recent BatchID in the log and reverts those rows.
    """
    if log_df is None or len(log_df) == 0:
        raise ValueError("Schedule_Log is empty. Nothing to undo.")

    # Find last batch by newest GeneratedAtUTC among rows that have BatchID
    df = log_df.copy()
    df["BatchID"] = df["Notes"].apply(lambda n: parse_note_field(n, "BatchID") or "")
    df = df[df["BatchID"] != ""].copy()
    if len(df) == 0:
        raise ValueError("No BatchID entries found in Notes. Cannot safely undo.")

    # Newest by GeneratedAtUTC (lexicographic works for ISO)
    df = df.sort_values("GeneratedAtUTC", ascending=False)
    last_batch = df.iloc[0]["BatchID"]

    batch_rows = log_df[log_df["Notes"].astype(str).str.contains(f"BatchID={last_batch}", na=False)].copy()
    if len(batch_rows) == 0:
        raise ValueError("Last batch could not be located.")

    updated_roster = roster_df.copy()

    # Revert roster per row using Notes
    for _, r in batch_rows.iterrows():
        name = str(r["Name"]).strip()
        notes = str(r["Notes"])

        delta_str = parse_note_field(notes, "Delta")
        prev_tc_str = parse_note_field(notes, "PrevTrueCount")
        prev_lr = parse_note_field(notes, "PrevLastRun") or ""

        if delta_str is None or prev_tc_str is None:
            raise ValueError("Cannot undo because Notes are missing Delta or PrevTrueCount.")

        prev_tc = int(prev_tc_str)

        idx = updated_roster.index[updated_roster["Name"] == name]
        if len(idx) != 1:
            raise ValueError(f"Roster row not found or duplicated for name: {name}")
        i = idx[0]

        updated_roster.at[i, "TrueCount"] = int(prev_tc)
        updated_roster.at[i, "LastRun"] = str(prev_lr).strip()

    # Remove the batch rows from log
    updated_log = log_df[~log_df["Notes"].astype(str).str.contains(f"BatchID={last_batch}", na=False)].copy()

    # Write back
    conn.update(worksheet=roster_ws, data=updated_roster[ROSTER_COLS])
    conn.update(worksheet=log_ws, data=updated_log[LOG_COLS])

    return updated_roster, updated_log


# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Sheets")
    roster_ws = st.text_input("Roster worksheet (tab name)", value=DEFAULT_ROSTER_WS)
    log_ws = st.text_input("Schedule_Log worksheet (tab name)", value=DEFAULT_LOG_WS)

    st.divider()
    st.header("Run")
    service_date = st.date_input("Service date", value=date.today())

    st.divider()
    st.header("Cameras active")
    active_roles = []
    for cam in CAMERAS_ALL:
        if st.checkbox(cam, value=True, key=f"active_{cam}"):
            active_roles.append(cam)

    st.divider()
    st.header("Role scoring weights (edit anytime)")
    st.caption("These weights drive which people are 'due' for premium roles.")
    role_values: Dict[str, int] = {}
    for cam in CAMERAS_ALL:
        default_val = {"Cam1": -2, "Cam2": -2, "Cam3": 0, "Cam4": 2, "Cam5": 4, "Cam6": 5}.get(cam, 0)
        role_values[cam] = int(st.number_input(f"{cam} value", value=int(default_val), step=1, key=f"val_{cam}"))

    st.divider()
    st.header("TrueCount deltas on commit")
    st.caption("This is the actual TrueCount change applied after committing.")
    role_truecount_delta: Dict[str, int] = {}
    for cam in CAMERAS_ALL:
        default_delta = {"Cam1": +1, "Cam2": +1, "Cam3": 0, "Cam4": -1, "Cam5": -2, "Cam6": -3}.get(cam, 0)
        role_truecount_delta[cam] = int(st.number_input(f"{cam} delta", value=int(default_delta), step=1, key=f"delta_{cam}"))

    st.divider()
    st.header("Rules")
    hard_avoid = st.checkbox("Hard block AvoidCameras", value=True)
    hard_back_to_back = st.checkbox("Hard block back-to-back same camera", value=False)
    enable_coverage = st.checkbox("Enable coverage rule", value=True)
    coverage_window_days = int(st.number_input("Coverage window (serving days)", value=10, step=1, min_value=1))

    preferred_bonus = int(st.number_input("Preferred bonus", value=20, step=1))
    avoid_penalty = int(st.number_input("Avoid penalty", value=50, step=1))
    back_to_back_penalty = int(st.number_input("Back-to-back penalty", value=40, step=1))
    coverage_penalty = int(st.number_input("Coverage repeat penalty", value=10, step=1))

    st.divider()
    st.header("Generate")
    n_options = int(st.number_input("How many options", value=5, min_value=1, max_value=10, step=1))
    n_random_samples = int(st.number_input("Random samples (only if needed)", value=500, min_value=50, max_value=10000, step=50))


# =========================
# Load Data
# =========================
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Roster (from Google Sheet)")
    try:
        roster = load_roster(roster_ws)
        st.dataframe(roster[ROSTER_COLS], use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.stop()

with colB:
    st.subheader("Schedule Log (latest)")
    try:
        schedule_log = load_log(log_ws)
        if len(schedule_log) > 0:
            st.dataframe(schedule_log.tail(12), use_container_width=True)
        else:
            st.info("Schedule_Log is empty (that is OK).")
    except Exception as e:
        st.error(str(e))
        st.stop()


# =========================
# Due List Panel
# =========================
st.divider()
st.subheader("Due list (highest TrueCount)")
due = roster[["Name", "TrueCount", "LastRun", "PreferredCameras", "AvoidCameras"]].copy()
due = due.sort_values("TrueCount", ascending=False)
st.dataframe(due, use_container_width=True)


# =========================
# Team Selection + Locks + Generate
# =========================
st.divider()
st.subheader("Build today's team, generate options, commit one")

if len(active_roles) == 0:
    st.error("Select at least one camera role in the sidebar.")
    st.stop()

team_names_all = roster["Name"].tolist()
team_selected = st.multiselect(
    "Select today's team (must match number of active cameras)",
    options=team_names_all,
    default=[],
)

if len(team_selected) != len(active_roles):
    st.warning(f"Select exactly {len(active_roles)} people (you selected {len(team_selected)}).")
    st.stop()

team_df = roster[roster["Name"].isin(team_selected)].copy()

# Coverage state
coverage_map = build_coverage_state(schedule_log, window_serving_days=coverage_window_days)

st.markdown("### Locks + Admin overrides (optional)")
lock_cols = st.columns(3)
locks: Dict[str, str] = {}
with lock_cols[0]:
    lock_role = st.selectbox("Lock role (optional)", options=["(none)"] + active_roles, index=0)
with lock_cols[1]:
    lock_name = st.selectbox("Lock person to that role", options=["(none)"] + team_selected, index=0)
with lock_cols[2]:
    add_lock = st.button("Add lock")

if "locks" not in st.session_state:
    st.session_state["locks"] = {}

if add_lock and lock_role != "(none)" and lock_name != "(none)":
    st.session_state["locks"][lock_role] = lock_name

if st.button("Clear all locks"):
    st.session_state["locks"] = {}

locks = dict(st.session_state["locks"])
if locks:
    st.write("Current locks:")
    st.json(locks)

st.markdown("### Generate options")
gen = st.button("Generate 3–5 options", type="primary")

if gen:
    try:
        options = generate_options(
            team_df=team_df,
            active_roles=active_roles,
            role_values=role_values,
            preferred_bonus=preferred_bonus,
            avoid_penalty=avoid_penalty,
            back_to_back_penalty=back_to_back_penalty,
            coverage_penalty=coverage_penalty,
            hard_avoid=hard_avoid,
            hard_back_to_back=hard_back_to_back,
            enable_coverage=enable_coverage,
            coverage_map=coverage_map,
            locks=locks,
            n_options=n_options,
            n_random_samples=n_random_samples,
        )
        st.session_state["options"] = options
        st.success(f"Generated {len(options)} option(s).")
    except Exception as e:
        st.error(str(e))

if "options" in st.session_state:
    options = st.session_state["options"]
    st.markdown("### Options")
    for i, opt in enumerate(options, start=1):
        with st.expander(f"Option {i}  |  Score: {opt['score']}", expanded=(i == 1)):
            df_opt = pd.DataFrame(opt["assignments"])
            st.dataframe(df_opt, use_container_width=True)
            st.markdown("**Why this option**")
            for line in opt["why"][:30]:
                st.write("- " + line)

    st.markdown("### Commit one option")
    option_index = st.number_input("Option number to commit", min_value=1, max_value=len(options), value=1, step=1)
    commit_clicked = st.button("Commit selected option to Google Sheet", type="secondary")

    st.markdown("### Undo last commit")
    undo_clicked = st.button("Undo last committed schedule", type="secondary")

    if commit_clicked:
        try:
            chosen = options[int(option_index) - 1]
            batch_id = f"{utc_now_iso()}_{random.randint(1000,9999)}"
            updated_roster, updated_log = apply_commit(
                roster_df=roster,
                log_df=schedule_log,
                roster_ws=roster_ws,
                log_ws=log_ws,
                service_date=service_date,
                commit_option=chosen,
                role_truecount_delta=role_truecount_delta,
                batch_id=batch_id,
            )
            st.success(f"Committed. BatchID stored in Notes: {batch_id}")

            # Refresh in-session views
            roster = updated_roster
            schedule_log = updated_log
            st.session_state.pop("options", None)
        except Exception as e:
            st.error(str(e))

    if undo_clicked:
        try:
            updated_roster, updated_log = undo_last_commit(
                roster_df=roster,
                log_df=schedule_log,
                roster_ws=roster_ws,
                log_ws=log_ws,
            )
            st.success("Undid last commit (TrueCount + LastRun reverted, log rows removed).")

            roster = updated_roster
            schedule_log = updated_log
            st.session_state.pop("options", None)
        except Exception as e:
            st.error(str(e))
