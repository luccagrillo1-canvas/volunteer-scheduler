import streamlit as st
import pandas as pd
from datetime import date
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Volunteer Scheduler V1", layout="wide")
st.title("Volunteer Scheduler V1 (Blackjack Fairness)")

# Connect to Google Sheets (Streamlit Secrets must define connections.gsheets)
conn = st.connection("gsheets", type=GSheetsConnection)

DEFAULT_WORKSHEET = "Roster"

with st.sidebar:
    st.header("Settings")
    worksheet = st.text_input("Worksheet (tab name)", value=DEFAULT_WORKSHEET)
    lead_keyword = st.text_input("Lead capability keyword", value="lead")
    service_date = st.date_input("Service date", value=date.today())

def parse_caps(x: str) -> set[str]:
    if pd.isna(x):
        return set()
    return {c.strip().lower() for c in str(x).split(",") if c.strip()}

def load_roster() -> pd.DataFrame:
    df = conn.read(worksheet=worksheet)
    # Normalize columns
    required = ["Name", "Role_Capability", "Fairness_Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet: {missing}. Expected {required}")

    # CHANGED: fill blanks so NaNs don't become the string "nan"
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    df["Role_Capability"] = df["Role_Capability"].fillna("").astype(str)

    df["Fairness_Score"] = pd.to_numeric(df["Fairness_Score"], errors="coerce").fillna(0).astype(int)
    return df

def generate_schedule(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Determine who is eligible to be lead
    caps = df["Role_Capability"].apply(parse_caps)
    is_lead_eligible = caps.apply(lambda s: lead_keyword.lower() in s)

    lead_pool = df[is_lead_eligible].copy()
    grip_pool = df.copy()

    if len(lead_pool) < 1:
        raise ValueError("No one is eligible for Lead based on the keyword. Update Role_Capability or the keyword.")

    # Sort by Fairness_Score descending (highest credit first)
    lead_pool = lead_pool.sort_values("Fairness_Score", ascending=False)
    grip_pool = grip_pool.sort_values("Fairness_Score", ascending=False)

    lead = lead_pool.iloc[0]
    # Remove lead from grip pool
    grip_pool = grip_pool[grip_pool["Name"] != lead["Name"]]

    if len(grip_pool) < 3:
        raise ValueError("Not enough people to fill 3 Grip spots after choosing Lead.")

    grips = grip_pool.iloc[:3]

    schedule = pd.DataFrame(
        [
            {"Date": service_date, "Role": "Lead Camera", "Name": lead["Name"]},
            {"Date": service_date, "Role": "Grip", "Name": grips.iloc[0]["Name"]},
            {"Date": service_date, "Role": "Grip", "Name": grips.iloc[1]["Name"]},
            {"Date": service_date, "Role": "Grip", "Name": grips.iloc[2]["Name"]},
        ]
    )

    # Update scores in a copy of df
    updated = df.copy()
    # Lead: subtract 2
    updated.loc[updated["Name"] == lead["Name"], "Fairness_Score"] -= 2
    # Grips: add 1
    for g in grips["Name"].tolist():
        updated.loc[updated["Name"] == g, "Fairness_Score"] += 1

    return schedule, updated

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Roster (from Google Sheet)")
    try:
        roster = load_roster()
        st.dataframe(roster, use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.stop()

with col2:
    st.subheader("Generate Schedule")
    if st.button("Generate Schedule", type="primary"):
        try:
            schedule_df, updated_roster = generate_schedule(roster)
            st.success("Schedule generated.")
            st.write("### Schedule")
            st.dataframe(schedule_df, use_container_width=True)

            st.write("### Updated Fairness Scores (not saved yet)")
            st.dataframe(updated_roster.sort_values("Fairness_Score", ascending=False), use_container_width=True)

            st.session_state["updated_roster"] = updated_roster
        except Exception as e:
            st.error(str(e))

st.divider()
st.subheader("Save updated scores back to Google Sheet")

if "updated_roster" in st.session_state:
    if st.button("Commit Updated Roster to Sheet"):
        try:
            conn.update(worksheet=worksheet, data=st.session_state["updated_roster"])
            st.success("Saved to Google Sheet.")
        except Exception as e:
            st.error(
                "Could not write back to the sheet. This usually means you need a service account / editor access.\n\n"
                f"Error: {e}"
            )
else:
    st.info("Generate a schedule first, then you can commit the updated scores.")
