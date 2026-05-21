import streamlit as st
import pandas as pd
import joblib

# 1. Load the pre-trained assets AND the new metadata
model = joblib.load('refund_liability_model.pkl')
expected_columns = joblib.load('model_columns.pkl')
metadata = joblib.load('app_metadata.pkl') 

# Configure page to use full width and hide the sidebar by default
st.set_page_config(page_title="Disruption Liability Engine", layout="wide", initial_sidebar_state="collapsed")

# Header Section
st.title("🚆 Disruption Liability Engine")
st.markdown("Predict financial exposure and refund probabilities for delayed passengers in real-time.")
st.markdown("---")

# 2. Main Input Canvas (Grouped by Operational Logic)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Route & Disruption Data")
    
    # Cascading Station Logic
    departure_station = st.selectbox("Departure Station", list(metadata['routes'].keys()))
    arrival_destination = st.selectbox("Arrival Destination", metadata['routes'][departure_station])
    
    # Time Variables
    delay_minutes = st.slider("Current Delay (Minutes)", min_value=1, max_value=120, value=30)
    departure_hour = st.slider("Departure Hour (24H)", min_value=0, max_value=23, value=8)

with col2:
    st.subheader("🎟️ Passenger & Ticket Profile")
    
    # Group ticket details horizontally within the column
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        ticket_class = st.radio("Ticket Class", ["Standard", "First Class"])
        ticket_type = st.selectbox("Ticket Type", ["Advance", "Anytime", "Off-Peak"])
    with t_col2:
        purchase_type = st.radio("Purchase Type", ["Online", "Station"])
        railcard = st.selectbox("Railcard Used", metadata['railcards'])
    
    booking_window = st.number_input("Booking Window (Days in Advance)", min_value=0, value=7)
    
    # Dynamic Price Calculation Logic
    lookup_key = (departure_station, arrival_destination, ticket_class, ticket_type, railcard)
    historical_price = metadata['pricing'].get(lookup_key, 50.0)
    price = st.number_input("Ticket Price (£) - Auto-filled by Historical Median", min_value=1.0, value=float(historical_price))

st.markdown("---")

# 3. The Execution Engine
# use_container_width makes the button massive and unmissable
if st.button("Calculate Financial Liability", type="primary", use_container_width=True):
    
    # Reconstruct the Route format the model expects
    model_route = f"{departure_station} to {arrival_destination}"
    
    input_data = pd.DataFrame({
        'Price': [price],
        'Delay Minutes': [delay_minutes],
        'Departure Hour': [departure_hour],
        'Booking Window (Days)': [booking_window],
        'Purchase Type': [purchase_type],
        'Railcard': [railcard],
        'Ticket Class': [ticket_class],
        'Ticket Type': [ticket_type],
        'Route': [model_route] 
    })
    
    # Apply One-Hot Encoding and align columns
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Generate Prediction
    probability = model.predict_proba(input_encoded)[0][1]
    
    # Calculate Impact
    refund_multiplier = 0.5 if delay_minutes >= 30 else 0.25 
    predicted_liability = price * refund_multiplier * probability
    
    # 4. Dynamic Visual Output System
    st.subheader("Actionable Intelligence")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Predicted Claim Probability", value=f"{probability * 100:.1f}%")
        # Visual Progress Bar for immediate risk assessment
        st.progress(float(probability))
        
    with res_col2:
        st.metric(label="Projected Financial Liability", value=f"£{predicted_liability:.2f}")
    
    # Operational Alert Logic
    if probability >= 0.75:
        st.error("🚨 CRITICAL RISK: High probability of financial leakage. Immediate operational triage required.")
    elif probability >= 0.40:
        st.warning("⚠️ MODERATE RISK: Passenger is approaching the friction threshold for a claim.")
    else:
        st.success("✅ LOW RISK: Passenger is unlikely to execute a claims process at this disruption level.")