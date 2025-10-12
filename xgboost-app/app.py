import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="CLARION: Churn Learning with AI-driven Reasoning and InterpretatiON",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Minimal Ultramarine Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    body, .main {
        background-color: #0a0e27;
        color: #E0E0E0;
    }

    h1, h2, h3 {
        color: #FFFFFF;
        letter-spacing: -0.3px;
        font-weight: 600;
    }
    h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.2rem; margin-top: 1.2rem; border-bottom: 1px solid #1a2a8f; padding-bottom: 0.4rem; }
    h3 { font-size: 1rem; margin: 0.8rem 0 0.3rem 0; }

    p, label, span, .stMarkdown { color: #B8B8B8 !important; }

    section[data-testid="stSidebar"] {
        background-color: #0b102f;
        border-right: 1px solid #1a2a8f;
    }

    /* --- Components --- */
    .metric-card {
        background-color: #10173a;
        border: 1px solid #1a2a8f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 { font-size: 0.8rem; color: #b8b8b8; text-transform: uppercase; margin-bottom: 0.2rem; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; }

    .info-box {
        background-color: #10173a;
        border-left: 3px solid #1a2a8f;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background-color: #1a2a8f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.4rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #263fe0;
        transform: translateY(-1px);
    }

    .churn-card, .no-churn-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .churn-card { background-color: #9b1c1c; }
    .no-churn-card { background-color: #1a2a8f; }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model_and_explainer():
    try:
        model = joblib.load('models/xgb_clean.joblib')
        explainer = joblib.load('models/shap_explainer_clean.joblib')
        return model, explainer
    except FileNotFoundError:
        st.error("Model / Explainer files not found.")
        return None, None

model, explainer = load_model_and_explainer()

# --- Main App ---
def main():
    st.title("CLARION: Churn Learning with AI-driven Reasoning and InterpretatiON")
    st.markdown('<div class="info-box">Upload <code>test.csv</code> to view churn predictions and interpretability results.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your test data:", type=["csv"])

    if uploaded_file and model and explainer:
        try:
            test_df = pd.read_csv(uploaded_file)
            X_test = test_df.copy()

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            shap_values = explainer.shap_values(X_test)

            churn_count = sum(predictions)
            churn_rate = (churn_count / len(test_df)) * 100
            avg_prob = probabilities.mean() * 100

            # --- KPI Cards ---
            col1, col2, col3, col4 = st.columns(4)
            metrics = [
                ("Total Customers", len(test_df)),
                ("Predicted Churns", churn_count),
                ("Churn Rate", f"{churn_rate:.1f}%"),
                ("Avg Churn Probability", f"{avg_prob:.1f}%")
            ]
            for col, (label, val) in zip([col1, col2, col3, col4], metrics):
                col.markdown(f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <div class="value">{val}</div>
                </div>""", unsafe_allow_html=True)

            # --- Layout ---
            main_col, sidebar_col = st.columns([2.5, 1])

            # --- Global Feature Importance ---
            with main_col:
                st.header("Global Feature Importance")

                fig_summary, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False, plot_size=None)

                # --- Style main axis ---
                ax.set_xlabel("SHAP Value", color='white', fontsize=10)
                ax.set_ylabel("", color='white', fontsize=10)
                ax.tick_params(axis='x', colors='white', labelsize=9)
                ax.tick_params(axis='y', colors='white', labelsize=9)
                ax.set_facecolor("#10173a")
                fig_summary.patch.set_facecolor("#0a0e27")
                for spine in ax.spines.values():
                    spine.set_color("#1a2a8f")

                # --- Style colorbar text ---
                for cb_ax in fig_summary.axes:
                    if cb_ax is not ax:  # the colorbar axis
                        cb_ax.tick_params(colors='white', labelsize=9)
                        cb_ax.yaxis.label.set_color('white')
                        for spine in cb_ax.spines.values():
                            spine.set_color("#1a2a8f")

                plt.tight_layout()
                st.pyplot(fig_summary, use_container_width=True)



            with sidebar_col:
                st.header("Customer Analysis")
                customer_index = st.selectbox("Select a Customer", test_df.index)
                if customer_index is not None:
                    is_churn = predictions[customer_index] == 1
                    prob_value = probabilities[customer_index]

                    card_class = "churn-card" if is_churn else "no-churn-card"
                    churn_text = "Churn" if is_churn else "Won't Churn"

                    st.markdown(f'<div class="{card_class}">{churn_text}</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Probability of Churn</h3>
                        <div class="value">{prob_value:.1%}</div>
                    </div>""", unsafe_allow_html=True)

                    st.subheader("Top Feature Impacts")

                    shap_vals = shap_values[customer_index]
                    feature_names = X_test.columns
                    indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
                    top_shap_vals = shap_vals[indices]
                    top_features = feature_names[indices]

                    colors = ['#e74c3c' if v > 0 else "#3647b9" for v in top_shap_vals]
                    fig_single, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(range(len(top_shap_vals)), top_shap_vals, color=colors)
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features, fontsize=9, color='white')
                    ax.set_xlabel('SHAP Value', fontsize=9, color='white')
                    ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
                    ax.tick_params(axis='x', colors='white', labelsize=8)
                    ax.set_facecolor("#10173a")
                    fig_single.patch.set_facecolor("#0a0e27")
                    plt.tight_layout()
                    st.pyplot(fig_single, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ensure your CSV has correct columns and format.")
    elif not (model and explainer):
        st.warning("Model could not be loaded.")
    else:
        st.info("Upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()
