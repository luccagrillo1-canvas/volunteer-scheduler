"""
Camera Operator Scheduler V2
Fairness-based volunteer scheduling with weighted camera positions
"""

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime, timezone
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

# Page configuration
st.set_page_config(
    page_title="Camera Scheduler V2",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Camera weights (higher = more desirable)
CAMERA_WEIGHTS = {
    "Cam1": 1,
    "Cam2": 1,
    "Cam3": 2,
    "Cam4": 2,
    "Cam5": 3,
    "Cam6": 4
}

CAMERAS = list(CAMERA_WEIGHTS.keys())
COVERAGE_WINDOW = 10  # Days to track camera coverage

# Google Sheets setup
@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client with credentials from Streamlit secrets"""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Failed to initialize Google Sheets: {e}")
        st.stop()

def get_spreadsheet():
    """Get the main spreadsheet"""
    try:
        client = get_google_sheets_client()
        sheet_url = st.secrets.get("sheet_url", "")
        if not sheet_url:
            st.error("Sheet URL not found in secrets. Please add 'sheet_url' to your Streamlit secrets.")
            st.stop()
        return client.open_by_url(sheet_url)
    except Exception as e:
        st.error(f"Failed to open spreadsheet: {e}")
        st.stop()

def load_roster() -> pd.DataFrame:
    """Load roster data from Google Sheet"""
    try:
        spreadsheet = get_spreadsheet()
        roster_sheet = spreadsheet.worksheet("Roster")
        data = roster_sheet.get_all_records()
        df = pd.DataFrame(data)

        # Ensure required columns exist
        required_cols = ["Name", "RoleCapability", "TrueCount", "Email",
                        "SkillLevel", "MedicalRestrictions", "PreferredCameras",
                        "AvoidCameras", "LastRun"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        # Convert TrueCount to numeric
        df["TrueCount"] = pd.to_numeric(df["TrueCount"], errors="coerce").fillna(0)

        return df
    except Exception as e:
        st.error(f"Failed to load roster: {e}")
        return pd.DataFrame()

def load_schedule_log() -> pd.DataFrame:
    """Load schedule log from Google Sheet"""
    try:
        spreadsheet = get_spreadsheet()
        try:
            log_sheet = spreadsheet.worksheet("Schedule_Log")
        except gspread.exceptions.WorksheetNotFound:
            # Create Schedule_Log if it doesn't exist
            log_sheet = spreadsheet.add_worksheet("Schedule_Log", rows=100, cols=10)
            log_sheet.update('A1:E1', [["Date", "Name", "RoleAssigned", "GeneratedAtUTC", "Notes"]])
            return pd.DataFrame(columns=["Date", "Name", "RoleAssigned", "GeneratedAtUTC", "Notes"])

        data = log_sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to load schedule log: {e}")
        return pd.DataFrame()

def save_roster(df: pd.DataFrame):
    """Save roster data back to Google Sheet"""
    try:
        spreadsheet = get_spreadsheet()
        roster_sheet = spreadsheet.worksheet("Roster")

        # Convert dataframe to list of lists
        values = [df.columns.tolist()] + df.values.tolist()
        roster_sheet.clear()
        roster_sheet.update('A1', values)

        st.success("✅ Roster updated successfully!")
    except Exception as e:
        st.error(f"Failed to save roster: {e}")

def append_to_log(assignments: Dict[str, str], service_date: str, notes: str = ""):
    """Append new assignments to schedule log"""
    try:
        spreadsheet = get_spreadsheet()
        log_sheet = spreadsheet.worksheet("Schedule_Log")

        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for person, camera in assignments.items():
            rows.append([service_date, person, camera, timestamp, notes])

        if rows:
            log_sheet.append_rows(rows)
            st.success(f"✅ Logged {len(rows)} assignments")
    except Exception as e:
        st.error(f"Failed to append to log: {e}")

def get_recent_assignments(person: str, log_df: pd.DataFrame, window: int = COVERAGE_WINDOW) -> List[str]:
    """Get recent camera assignments for a person"""
    person_log = log_df[log_df["Name"] == person].tail(window)
    return person_log["RoleAssigned"].tolist()

def calculate_coverage_score(person: str, camera: str, log_df: pd.DataFrame) -> float:
    """
    Calculate coverage score - higher score for cameras not recently used
    Returns: 0-1, where 1 = hasn't used this camera recently
    """
    recent = get_recent_assignments(person, log_df, COVERAGE_WINDOW)
    if not recent:
        return 1.0

    # Count how many times they've used this camera recently
    count = recent.count(camera)
    # Return inverse - higher score if less usage
    return max(0, 1 - (count / len(recent)))

def calculate_back_to_back_penalty(person: str, camera: str, log_df: pd.DataFrame) -> float:
    """
    Calculate penalty for back-to-back same camera assignment
    Returns: 0-1, where 0 = just used this camera, 1 = no recent usage
    """
    if log_df.empty:
        return 1.0

    person_log = log_df[log_df["Name"] == person]
    if person_log.empty:
        return 1.0

    last_camera = person_log.iloc[-1]["RoleAssigned"]
    if last_camera == camera:
        return 0.0  # Heavy penalty for same camera
    else:
        return 1.0  # No penalty

def parse_camera_list(camera_str: str) -> List[str]:
    """Parse comma-separated camera list"""
    if not camera_str or pd.isna(camera_str):
        return []
    return [cam.strip() for cam in str(camera_str).split(",") if cam.strip()]

def calculate_preference_score(person_data: pd.Series, camera: str) -> float:
    """
    Calculate preference score
    Returns: 0-1.5, where >1 = preferred, <1 = avoided
    """
    preferred = parse_camera_list(person_data.get("PreferredCameras", ""))
    avoided = parse_camera_list(person_data.get("AvoidCameras", ""))

    if camera in preferred:
        return 1.3  # Bonus for preferred
    elif camera in avoided:
        return 0.7  # Penalty for avoided
    else:
        return 1.0  # Neutral

def generate_schedule_option(
    available_people: List[str],
    roster_df: pd.DataFrame,
    log_df: pd.DataFrame,
    locked_assignments: Dict[str, str] = None
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Generate one schedule option using weighted scoring
    Returns: (assignments, scoring_details)
    """
    locked_assignments = locked_assignments or {}
    assignments = locked_assignments.copy()
    remaining_cameras = [cam for cam in CAMERAS if cam not in assignments.values()]
    remaining_people = [p for p in available_people if p not in assignments.keys()]

    scoring_details = {}

    # Build scoring matrix
    scores = defaultdict(dict)
    for person in remaining_people:
        person_data = roster_df[roster_df["Name"] == person].iloc[0]
        true_count = person_data["TrueCount"]

        for camera in remaining_cameras:
            # Calculate component scores
            weight = CAMERA_WEIGHTS[camera]
            fairness_score = 1.0 / (true_count + 1)  # Lower TrueCount = higher priority
            coverage_score = calculate_coverage_score(person, camera, log_df)
            back_to_back_score = calculate_back_to_back_penalty(person, camera, log_df)
            preference_score = calculate_preference_score(person_data, camera)

            # Combined score with weights
            total_score = (
                fairness_score * 2.0 +      # Fairness is most important
                coverage_score * 1.5 +       # Coverage is important
                back_to_back_score * 1.5 +   # Avoid back-to-back
                preference_score * 1.0 +     # Honor preferences
                (weight * 0.3)               # Give high cameras slightly to those who deserve it
            )

            # Add small random factor to break ties
            total_score += random.random() * 0.1

            scores[person][camera] = total_score

            # Store details for explanation
            scoring_details[f"{person}-{camera}"] = {
                "fairness": fairness_score,
                "coverage": coverage_score,
                "back_to_back": back_to_back_score,
                "preference": preference_score,
                "camera_weight": weight,
                "total": total_score
            }

    # Greedy assignment - assign highest scoring person-camera pairs
    while remaining_people and remaining_cameras:
        # Find best assignment
        best_person = None
        best_camera = None
        best_score = -1

        for person in remaining_people:
            for camera in remaining_cameras:
                if scores[person][camera] > best_score:
                    best_score = scores[person][camera]
                    best_person = person
                    best_camera = camera

        if best_person and best_camera:
            assignments[best_person] = best_camera
            remaining_people.remove(best_person)
            remaining_cameras.remove(best_camera)
        else:
            break

    return assignments, scoring_details

def explain_assignment(person: str, camera: str, scoring_details: Dict, roster_df: pd.DataFrame) -> str:
    """Generate human-readable explanation for an assignment"""
    key = f"{person}-{camera}"
    if key not in scoring_details:
        return "N/A"

    details = scoring_details[key]
    person_data = roster_df[roster_df["Name"] == person].iloc[0]
    true_count = person_data["TrueCount"]

    reasons = []

    # Fairness
    if details["fairness"] > 0.5:
        reasons.append(f"Low TrueCount ({true_count:.1f}) - due for better cameras")

    # Coverage
    if details["coverage"] > 0.7:
        reasons.append(f"Hasn't run {camera} recently")

    # Back-to-back
    if details["back_to_back"] == 0:
        reasons.append(f"⚠️ Just ran {camera} last time")

    # Preference
    preferred = parse_camera_list(person_data.get("PreferredCameras", ""))
    avoided = parse_camera_list(person_data.get("AvoidCameras", ""))
    if camera in preferred:
        reasons.append(f"✅ Preferred camera")
    elif camera in avoided:
        reasons.append(f"⚠️ Avoided camera")

    # Camera weight
    weight = CAMERA_WEIGHTS[camera]
    if weight >= 3:
        reasons.append(f"Premium camera (weight {weight})")

    return " | ".join(reasons) if reasons else "Standard assignment"

def calculate_option_summary(assignments: Dict[str, str], roster_df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for a schedule option"""
    total_weight = sum(CAMERA_WEIGHTS[cam] for cam in assignments.values())
    avg_true_count = roster_df[roster_df["Name"].isin(assignments.keys())]["TrueCount"].mean()

    premium_assignments = sum(1 for cam in assignments.values() if CAMERA_WEIGHTS[cam] >= 3)

    return {
        "total_weight": total_weight,
        "avg_true_count": avg_true_count,
        "premium_assignments": premium_assignments
    }

# ============= STREAMLIT UI =============

st.title("🎥 Camera Operator Scheduler V2")
st.markdown("**Fairness-based scheduling with weighted camera positions**")

# Load data
roster_df = load_roster()
log_df = load_schedule_log()

if roster_df.empty:
    st.error("⚠️ No roster data found. Please set up your Roster tab in Google Sheets.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    service_date = st.date_input("Service Date", datetime.now())

    st.markdown("---")
    st.subheader("📊 Camera Weights")
    for cam, weight in CAMERA_WEIGHTS.items():
        st.text(f"{cam}: {'⭐' * weight}")

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Generate Schedule",
    "👥 Due List",
    "⚙️ Admin Tools",
    "📊 Roster View",
    "📜 Schedule History"
])

# TAB 1: Generate Schedule
with tab1:
    st.header("Generate Schedule Options")

    # Team selection
    st.subheader("1️⃣ Select Team Members")
    all_names = sorted(roster_df["Name"].tolist())
    selected_team = st.multiselect(
        "Select 6 camera operators for this service:",
        options=all_names,
        default=all_names[:6] if len(all_names) >= 6 else all_names
    )

    if len(selected_team) != 6:
        st.warning(f"⚠️ Please select exactly 6 operators. Currently selected: {len(selected_team)}")

    # Locked assignments
    st.subheader("2️⃣ Lock Assignments (Optional)")
    st.caption("Lock specific people to cameras before generating options")

    locked_assignments = {}
    lock_cols = st.columns(3)
    for i in range(2):  # Allow up to 2 locks
        with lock_cols[i]:
            person = st.selectbox(
                f"Lock Person {i+1}",
                options=[""] + selected_team,
                key=f"lock_person_{i}"
            )
            if person:
                available_cameras = [c for c in CAMERAS if c not in locked_assignments.values()]
                camera = st.selectbox(
                    f"to Camera",
                    options=available_cameras,
                    key=f"lock_camera_{i}"
                )
                if camera:
                    locked_assignments[person] = camera

    if locked_assignments:
        st.info(f"🔒 Locked: {', '.join([f'{p}→{c}' for p, c in locked_assignments.items()])}")

    # Generate options
    st.subheader("3️⃣ Generate Options")
    num_options = st.slider("Number of options to generate:", 3, 5, 3)

    if st.button("🎲 Generate Schedule Options", type="primary", use_container_width=True):
        if len(selected_team) != 6:
            st.error("⚠️ Please select exactly 6 operators")
        else:
            with st.spinner("Generating optimal schedules..."):
                options = []
                for i in range(num_options):
                    assignments, scoring_details = generate_schedule_option(
                        selected_team,
                        roster_df,
                        log_df,
                        locked_assignments
                    )
                    summary = calculate_option_summary(assignments, roster_df)
                    options.append({
                        "assignments": assignments,
                        "scoring_details": scoring_details,
                        "summary": summary
                    })

                st.session_state.generated_options = options
                st.session_state.selected_option_idx = None
                st.success(f"✅ Generated {len(options)} schedule options!")

    # Display options
    if "generated_options" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Schedule Options")

        for idx, option in enumerate(st.session_state.generated_options):
            with st.expander(f"**Option {idx + 1}** - Total Weight: {option['summary']['total_weight']} | Premium: {option['summary']['premium_assignments']}/6", expanded=(idx == 0)):

                # Assignment table
                assignment_data = []
                for person, camera in sorted(option["assignments"].items(),
                                            key=lambda x: CAMERAS.index(x[1])):
                    person_data = roster_df[roster_df["Name"] == person].iloc[0]
                    true_count = person_data["TrueCount"]
                    explanation = explain_assignment(person, camera, option["scoring_details"], roster_df)

                    assignment_data.append({
                        "Camera": camera,
                        "Operator": person,
                        "TrueCount": f"{true_count:.1f}",
                        "Weight": CAMERA_WEIGHTS[camera],
                        "Why?": explanation
                    })

                df_display = pd.DataFrame(assignment_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # Select button
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"✅ Select Option {idx + 1}", key=f"select_{idx}", use_container_width=True):
                        st.session_state.selected_option_idx = idx
                        st.success(f"Selected Option {idx + 1}!")

        # Commit selected option
        if st.session_state.get("selected_option_idx") is not None:
            st.markdown("---")
            st.subheader("💾 Commit Schedule")

            selected_idx = st.session_state.selected_option_idx
            selected_option = st.session_state.generated_options[selected_idx]

            st.info(f"Ready to commit **Option {selected_idx + 1}**")

            commit_notes = st.text_input("Notes (optional):", placeholder="e.g., Sunday morning service")

            if st.button("💾 Commit to Schedule", type="primary", use_container_width=True):
                with st.spinner("Committing schedule..."):
                    # Update TrueCounts
                    for person, camera in selected_option["assignments"].items():
                        weight = CAMERA_WEIGHTS[camera]
                        roster_df.loc[roster_df["Name"] == person, "TrueCount"] += weight
                        roster_df.loc[roster_df["Name"] == person, "LastRun"] = str(service_date)

                    # Save roster
                    save_roster(roster_df)

                    # Log assignments
                    append_to_log(selected_option["assignments"], str(service_date), commit_notes)

                    # Clear session state
                    del st.session_state.generated_options
                    del st.session_state.selected_option_idx

                    st.success("🎉 Schedule committed successfully!")
                    st.balloons()
                    st.rerun()

# TAB 2: Due List
with tab2:
    st.header("📊 Due List - Premium Camera Priority")
    st.caption("Ranked by who deserves premium cameras (Cam5, Cam6) most")

    # Calculate "due score" for premium cameras
    due_data = []
    for _, person in roster_df.iterrows():
        name = person["Name"]
        true_count = person["TrueCount"]

        # Get recent premium camera usage
        recent = get_recent_assignments(name, log_df, COVERAGE_WINDOW)
        premium_count = sum(1 for cam in recent if CAMERA_WEIGHTS.get(cam, 0) >= 3)
        premium_ratio = premium_count / len(recent) if recent else 0

        # Calculate due score (lower TrueCount + less premium usage = higher score)
        due_score = (1 / (true_count + 1)) * 100 + (1 - premium_ratio) * 50

        due_data.append({
            "Rank": 0,
            "Name": name,
            "TrueCount": f"{true_count:.1f}",
            "Recent Premium": f"{premium_count}/{len(recent) if recent else 0}",
            "Due Score": f"{due_score:.1f}",
            "Last Served": person.get("LastRun", "Never")
        })

    # Sort by due score
    due_df = pd.DataFrame(due_data)
    due_df = due_df.sort_values("Due Score", ascending=False)
    due_df["Rank"] = range(1, len(due_df) + 1)

    # Display with color coding
    st.dataframe(
        due_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Due Score": st.column_config.NumberColumn("Due Score", format="%.1f")
        }
    )

    st.caption("💡 Higher Due Score = More deserving of premium cameras (Cam5, Cam6)")

# TAB 3: Admin Tools
with tab3:
    st.header("⚙️ Admin Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔙 Undo Last Schedule")
        st.caption("Revert TrueCount and remove last log entries")

        if not log_df.empty:
            last_date = log_df.iloc[-1]["Date"] if not log_df.empty else "N/A"
            last_count = len(log_df[log_df["Date"] == last_date]) if not log_df.empty else 0
            st.info(f"Last schedule: **{last_date}** ({last_count} assignments)")

            if st.button("🔙 Undo Last Schedule", use_container_width=True):
                if st.button("⚠️ Confirm Undo", use_container_width=True, type="primary"):
                    with st.spinner("Reverting changes..."):
                        # Get last schedule entries
                        last_entries = log_df[log_df["Date"] == last_date]

                        # Revert TrueCounts
                        for _, entry in last_entries.iterrows():
                            person = entry["Name"]
                            camera = entry["RoleAssigned"]
                            weight = CAMERA_WEIGHTS.get(camera, 0)
                            roster_df.loc[roster_df["Name"] == person, "TrueCount"] -= weight

                        # Save roster
                        save_roster(roster_df)

                        # Remove log entries
                        try:
                            spreadsheet = get_spreadsheet()
                            log_sheet = spreadsheet.worksheet("Schedule_Log")
                            # Delete last N rows
                            num_rows = len(last_entries)
                            current_rows = len(log_sheet.get_all_values())
                            if num_rows > 0:
                                log_sheet.delete_rows(current_rows - num_rows + 1, current_rows)
                            st.success(f"✅ Undone {num_rows} assignments from {last_date}")
                        except Exception as e:
                            st.error(f"Failed to remove log entries: {e}")

                        st.rerun()
        else:
            st.warning("No schedules to undo")

    with col2:
        st.subheader("👤 Admin Override")
        st.caption("Manually assign someone (updates TrueCount)")

        override_person = st.selectbox("Select Person:", roster_df["Name"].tolist())
        override_camera = st.selectbox("Assign to Camera:", CAMERAS)
        override_date = st.date_input("Service Date:", datetime.now(), key="override_date")

        if st.button("✅ Apply Override", use_container_width=True):
            with st.spinner("Applying override..."):
                weight = CAMERA_WEIGHTS[override_camera]
                roster_df.loc[roster_df["Name"] == override_person, "TrueCount"] += weight
                roster_df.loc[roster_df["Name"] == override_person, "LastRun"] = str(override_date)

                save_roster(roster_df)
                append_to_log({override_person: override_camera}, str(override_date), "Admin override")

                st.success(f"✅ {override_person} → {override_camera} (TrueCount +{weight})")
                st.rerun()

    st.markdown("---")

    # Full Reset
    st.subheader("🚨 Full Reset")
    st.caption("⚠️ Clears all TrueCounts and schedule history. Column headers remain intact.")

    if st.button("🚨 Full Reset (Danger Zone)", use_container_width=True):
        with st.expander("⚠️ Confirm Full Reset", expanded=True):
            st.error("This will reset ALL TrueCounts to 0 and clear the entire schedule history!")

            confirm_text = st.text_input("Type 'RESET' to confirm:")

            if confirm_text == "RESET":
                if st.button("🔥 CONFIRM FULL RESET", type="primary", use_container_width=True):
                    with st.spinner("Resetting all data..."):
                        # Reset TrueCounts
                        roster_df["TrueCount"] = 0
                        roster_df["LastRun"] = ""
                        save_roster(roster_df)

                        # Clear schedule log (keep headers)
                        try:
                            spreadsheet = get_spreadsheet()
                            log_sheet = spreadsheet.worksheet("Schedule_Log")
                            log_sheet.clear()
                            log_sheet.update('A1:E1', [["Date", "Name", "RoleAssigned", "GeneratedAtUTC", "Notes"]])
                            st.success("✅ Full reset completed!")
                        except Exception as e:
                            st.error(f"Failed to clear log: {e}")

                        st.rerun()

# TAB 4: Roster View
with tab4:
    st.header("👥 Roster Management")

    st.dataframe(
        roster_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "TrueCount": st.column_config.NumberColumn("TrueCount", format="%.1f"),
            "Email": st.column_config.TextColumn("Email"),
            "PreferredCameras": st.column_config.TextColumn("Preferred"),
            "AvoidCameras": st.column_config.TextColumn("Avoid")
        }
    )

    st.caption("💡 Edit data directly in Google Sheets, then click 'Refresh Data' in sidebar")

# TAB 5: Schedule History
with tab5:
    st.header("📜 Schedule History")

    if not log_df.empty:
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Schedules", len(log_df["Date"].unique()))
        with col2:
            st.metric("Total Assignments", len(log_df))
        with col3:
            last_date = log_df.iloc[-1]["Date"]
            st.metric("Last Schedule", last_date)

        st.markdown("---")

        # Filter by date range
        if len(log_df) > 0:
            date_filter = st.selectbox(
                "Filter by:",
                ["All History", "Last 5 Schedules", "Last 10 Schedules"]
            )

            if date_filter == "Last 5 Schedules":
                unique_dates = log_df["Date"].unique()[-5:]
                filtered_log = log_df[log_df["Date"].isin(unique_dates)]
            elif date_filter == "Last 10 Schedules":
                unique_dates = log_df["Date"].unique()[-10:]
                filtered_log = log_df[log_df["Date"].isin(unique_dates)]
            else:
                filtered_log = log_df

            st.dataframe(
                filtered_log,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No schedule history yet. Generate your first schedule in the 'Generate Schedule' tab!")

# Footer
st.markdown("---")
st.caption("🎥 Camera Scheduler V2 | Built with Streamlit | TrueCount Fairness System")
