import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Linear Regression CRISP-DM Workflow",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Aesthetics (Custom CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .crisp-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("⚙️ Simulation Settings")
st.sidebar.markdown("Adjust parameters to generate data and train the model.")

n_samples = st.sidebar.slider("Samples (n)", 100, 1000, 500)
true_a = st.sidebar.slider("True Slope (a)", -10.0, 10.0, 2.5, step=0.1)
true_b = st.sidebar.slider("True Intercept (b)", -50.0, 50.0, 10.0, step=0.5)
noise_mean = st.sidebar.slider("Noise Mean (μ)", -10.0, 10.0, 0.0, step=0.5)
noise_var = st.sidebar.slider("Noise Variance (σ²)", 0, 1000, 100)
seed = st.sidebar.number_input("Random Seed", value=42)

generate_btn = st.sidebar.button("🚀 Generate Data")

# --- Helper Functions ---
@st.cache_data
def generate_synthetic_data(n, a, b, n_mean, n_var, r_seed):
    np.random.seed(r_seed)
    x = np.random.uniform(-100, 100, n)
    noise = np.random.normal(n_mean, np.sqrt(n_var), n)
    y = a * x + b + noise
    return pd.DataFrame({'X': x, 'y': y})

# --- Initialize Session State ---
if 'data' not in st.session_state or generate_btn:
    with st.spinner("Generating data..."):
        st.session_state.data = generate_synthetic_data(n_samples, true_a, true_b, noise_mean, noise_var, seed)
        st.session_state.true_params = {'a': true_a, 'b': true_b}

df = st.session_state.data

# --- Main Title ---
st.title("📊 Linear Regression & CRISP-DM Workflow")
st.markdown("""
This application demonstrates a complete machine learning lifecycle using the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.
We use **Linear Regression** to model a synthetic linear relationship.
""")

# --- Phase 1: Business Understanding ---
st.markdown("<h2 class='crisp-header'>1. Business Understanding</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.write(r"""
    **Goal:** Develop a predictive model to estimate a continuous target variable $y$ based on a single feature $x$.
    **Problem Type:** Supervised Learning - Regression Analysis.
    **Success Criteria:**
    - High $R^2$ score (close to 1.0).
    - Low Root Mean Squared Error (RMSE).
    - Learned parameters $(\hat{a}, \hat{b})$ should be close to true parameters $(a, b)$.
    """)
with col2:
    st.info(r"💡 Linear Regression assumes: $y = ax + b + \epsilon$")

# --- Phase 2: Data Understanding ---
st.markdown("<h2 class='crisp-header'>2. Data Understanding</h2>", unsafe_allow_html=True)
col_stat1, col_stat2 = st.columns([1, 1])

with col_stat1:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with col_stat2:
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

st.subheader("Initial Data Distribution")
fig_dist, ax_dist = plt.subplots(1, 2, figsize=(12, 4))
ax_dist[0].hist(df['X'], bins=30, color='skyblue', edgecolor='black')
ax_dist[0].set_title("Distribution of Feature X")
ax_dist[1].hist(df['y'], bins=30, color='salmon', edgecolor='black')
ax_dist[1].set_title("Distribution of Target y")
st.pyplot(fig_dist)

# --- Phase 3: Data Preparation ---
st.markdown("<h2 class='crisp-header'>3. Data Preparation</h2>", unsafe_allow_html=True)
test_size = st.slider("Test Set Ratio", 0.1, 0.5, 0.2, step=0.05)

X = df[['X']]
y = df['y']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success(f"✅ Data split into: Training set ({len(X_train)} samples) and Test set ({len(X_test)} samples).")
st.write("Applied **StandardScaler** to normalize feature $X$.")

# --- Phase 4: Modeling ---
st.markdown("<h2 class='crisp-header'>4. Modeling</h2>", unsafe_allow_html=True)
with st.status("Training Linear Regression model...", expanded=True) as status:
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    time.sleep(0.5) # Simulate processing time
    status.update(label="Training Complete!", state="complete", expanded=False)

# Inverse transform parameters for comparison (scaling affects coefficients)
learned_a = model.coef_[0] / scaler.scale_[0]
learned_b = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])

st.write("Model trained successfully on scaled features.")

# --- Phase 5: Evaluation ---
st.markdown("<h2 class='crisp-header'>5. Evaluation</h2>", unsafe_allow_html=True)

# Predictions
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Metrics Panel
m1, m2, m3 = st.columns(3)
m1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
m3.metric("R² Score", f"{r2:.4f}")

# Visualize Results
st.subheader("Regression Visualization")
fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
ax_reg.scatter(X_test, y_test, alpha=0.5, label='Actual Data (Test set)', color='gray')

# Plot Regression Line
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_scaled = scaler.transform(x_range)
y_range_pred = model.predict(x_range_scaled)
ax_reg.plot(x_range, y_range_pred, color='red', linewidth=3, label='Regression Line')

ax_reg.set_xlabel("X")
ax_reg.set_ylabel("y")
ax_reg.legend()
st.pyplot(fig_reg)

# Parameter Comparison
st.subheader("Parameter Comparison")
p_col1, p_col2 = st.columns(2)
with p_col1:
    st.write("**True Parameters (Simulation)**")
    st.code(f"Slope (a): {st.session_state.true_params['a']}\nIntercept (b): {st.session_state.true_params['b']}")
with p_col2:
    st.write("**Learned Parameters (Estimated)**")
    st.code(f"Slope (â): {learned_a:.4f}\nIntercept (b̂): {learned_b:.4f}")

# --- Phase 6: Deployment ---
st.markdown("<h2 class='crisp-header'>6. Deployment</h2>", unsafe_allow_html=True)

tab_pred, tab_save = st.tabs(["🚀 Real-time Prediction", "💾 Model Export"])

with tab_pred:
    st.write("Enter $X$ value to predict $y$:")
    input_x = st.number_input("Input X", value=0.0)
    input_x_scaled = scaler.transform([[input_x]])
    pred_y = model.predict(input_x_scaled)[0]
    st.success(f"**Predicted y:** {pred_y:.4f}")
    st.info(f"Formula: $y \approx {learned_a:.4f}x + {learned_b:.4f}$")

with tab_save:
    st.write("Download the trained model and scaler for future use.")
    
    # Prepare data for download
    model_data = {
        'model': model,
        'scaler': scaler,
        'metadata': {
            'r2': r2,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    buffer = io.BytesIO()
    joblib.dump(model_data, buffer)
    st.download_button(
        label="📥 Download model.joblib",
        data=buffer.getvalue(),
        file_name="linear_regression_model.joblib",
        mime="application/octet-stream"
    )

st.divider()
st.caption("Powered by Streamlit & scikit-learn. A demo project for the CRISP-DM workflow.")
