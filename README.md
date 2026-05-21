# 🚆 Delay-Triggered Liability Engine: Machine Learning for Revenue Protection in UK Rail Logistics

An end-to-end XGBoost machine learning pipeline and interactive Streamlit application to predict passenger refund behavior and quantify real-time financial revenue leakage during transit delays.

---

## 1. The Strategic Business Problem
Transport logistics typically measure failure in minutes. This is a flawed operational metric. A 20-minute delay on a train full of £150 First Class tickets creates a massive financial liability, whereas a 40-minute delay on a train of £10 standard tickets creates very little. 

**The objective of this project is to shift crisis management from measuring *time* to predicting *financial liability*.**

By mapping the exact psychological and financial tipping points that cause a delayed passenger to demand a refund, this engine allows dispatchers to triage network disruptions based on hard balance-sheet impact.

---

## 2. Project Architecture & A-Z Flow
This project follows a strict machine learning lifecycle, moving from raw operational data to a live, interactive web application.

### Phase A: Data Isolation & Risk Mitigation
To train an accurate behavioral model, the dataset required aggressive filtering to prevent data leakage and mathematical paradoxes.
* **Dropped "Cancelled" Trains:** Cancellations carry a deterministic 100% legal liability. Including them creates a paradox where massive disruptions show low claim rates (due to abandoned claims), confusing the algorithm.
* **Dropped "On-Time" Trains:** Including these would bloat the dataset with irrelevant rows, allowing the model to artificially inflate its accuracy score by simply guessing "No Refund" on thousands of uninterrupted journeys.
* **Dropped "Reason for Delay":** Removed to prevent data leakage. Passengers do not know the exact operational cause of a delay while waiting; they only experience the lost time.

### Phase B: Feature Engineering
Raw strings were converted into numerical signals that capture human behavioral elasticity.
* **Continuous Delay Calculation:** Converted raw arrival strings into `Delay Minutes`, implementing programmatic logic to fix negative-time inversions for trains crossing the midnight threshold.
* **Time Binning:** Extracted `Departure Hour` to capture peak vs. off-peak behavioral shifts.
* **Dimensionality Reduction:** Dropped high-noise chronological artifacts like `Time of Purchase` to prevent the model from overfitting to random minute-level noise.

### Phase C: The XGBoost Predictive Model
The engine relies on an optimized XGBoost Classifier trained on the isolated disruption data.
* **Handling Imbalance:** The dataset contained heavily skewed refund claim rates. Implemented class-weight scaling (`scale_pos_weight`) to mathematically penalize the model for missing a valid refund claim.
* **Overfitting Prevention:** Restricted tree depth (`max_depth=4`) to prevent the model from memorizing the small 2.2K row dataset.
* **Performance:** The model prioritizes catching financial leakage, achieving **95% Recall** (successfully identifying 95% of all actual refund demands) and **90% Precision**.

### Phase D: Production Deployment (Streamlit)
The static machine learning model was deployed into a live, interactive dispatcher dashboard using Streamlit.
* **Decoupled Architecture:** The app imports the pre-trained XGBoost model (`.pkl`) to generate instant predictions without retraining.
* **Metadata Memory:** Extracted a cascading dictionary of valid routes and historical median pricing from the training set. This allows the app to dynamically auto-fill ticket prices and prevent users from inputting invalid station combinations.

---

## 3. The Live Application Dashboard
The Streamlit interface acts as a triage tool for transport dispatchers. 

**Inputs:**
* Geography (Cascading Departure and Arrival selections)
* Disruption Severity (Delay Minutes, Time of Day)
* Ticket Profile (Class, Type, Railcard, Purchase Method)

**Outputs:**
* **Predicted Claim Probability:** The exact percentage chance a specific passenger will endure the friction of the claims process.
* **Projected Financial Liability:** The monetary risk calculated by multiplying the ticket price by the legal refund tier and the behavioral probability.
* **Risk Alert System:** Visual triggers (Red/Yellow/Green) to categorize the immediate operational threat level.

---
