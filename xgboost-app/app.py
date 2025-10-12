import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CLARION: Churn Learning with AI-driven Reasoning and InterpretatiON",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0a0e27;
    }
    
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #FFFFFF;
        font-weight: 600;
        border-bottom: 2px solid #120a8f;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #E8E8E8;
        font-weight: 500;
    }
    
    p, .st-emotion-cache-16idsys p {
        color: #B8B8B8;
        line-height: 1.6;
    }
    
    .stMarkdown {
        background-color: #0f1535;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .uploadedFile {
        background-color: #13193a;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: #13193a;
        border-radius: 8px;
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #120a8f 0%, #1e3799 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(18, 10, 143, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card h3 {
        color: rgba(255, 255, 255, 0.85);
        margin: 0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: none;
    }
    
    .metric-card .value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .info-box {
        background-color: #0f1535;
        border-left: 4px solid #120a8f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #B8B8B8;
    }
    
    .info-box code {
        background-color: #13193a;
        padding: 2px 6px;
        border-radius: 4px;
        color: #FFFFFF;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0c1028;
        border-right: 1px solid #120a8f;
    }
    
    .stButton > button {
        background: linear-gradient(120deg, #120a8f 0%, #1e3799 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(18, 10, 143, 0.6);
    }
    
    .churn-card {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(192, 57, 43, 0.4);
    }
    
    .no-churn-card {
        background: linear-gradient(135deg, #120a8f 0%, #4169E1 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(18, 10, 143, 0.4);
    }
</style>
""", unsafe_allow_html=True)

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

def main():
    st.title("CLARION: Churn Learning with AI-driven Reasoning and InterpretatiON")

    st.markdown("""
    <div class="info-box">
    Upload <CODE>test.csv</CODE>.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your test data:", type=["csv"])

    if uploaded_file is not None and model is not None and explainer is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            X_test = test_df.copy()

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            shap_values = explainer.shap_values(X_test)

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Customers</h3>
                    <div class="value">{len(test_df)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                churn_count = sum(predictions)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Churns</h3>
                    <div class="value">{churn_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                churn_rate = (churn_count / len(test_df)) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Churn Rate</h3>
                    <div class="value">{churn_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_prob = probabilities.mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Churn Probability</h3>
                    <div class="value">{avg_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            main_col, sidebar_col = st.columns([2, 1])

            with main_col:
                st.header("Global Feature Importance")
                st.markdown("This plot shows the most significant features impacting churn predictions across all customers.")

                fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
                plt.title("Global Feature Importance", color='white', fontsize=16, pad=20)
                shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
                plt.gcf().set_facecolor("#0a0e27")
                plt.gca().set_facecolor("#0f1535")
                plt.gca().tick_params(axis='x', colors='white', labelsize=10)
                plt.gca().tick_params(axis='y', colors='white', labelsize=10)
                plt.gca().spines['bottom'].set_color('#120a8f')
                plt.gca().spines['left'].set_color('#120a8f')
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                st.pyplot(fig_summary)

            with sidebar_col:
                st.header("Customer Analysis")

                customer_index = st.selectbox(
                    "Select a Customer",
                    test_df.index,
                    key="customer_select"
                )

                if customer_index is not None:
                    is_churn = predictions[customer_index] == 1
                    churn_text = "Churn" if is_churn else "Not Churn"
                    prob_value = probabilities[customer_index]
                    
                    card_class = "churn-card" if is_churn else "no-churn-card"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>Churn Prediction</h3>
                        <div class="value">{churn_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Probability of Churn</h3>
                        <div class="value">{prob_value:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("Prediction Explanation")
                    st.markdown("Features pushing the prediction towards or away from churn.")
                    
                    fig_single = plt.figure(figsize=(10, 8))
                    
                    import numpy as np
                    shap_vals = shap_values[customer_index]
                    feature_names = X_test.columns
                    
                    # get top 10 features
                    indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
                    top_shap_vals = shap_vals[indices]
                    top_features = feature_names[indices]
                    
                    colors = ['#e74c3c' if val < 0 else '#120a8f' for val in top_shap_vals]
                    bars = plt.barh(range(len(top_shap_vals)), top_shap_vals, color=colors)
                    plt.yticks(range(len(top_shap_vals)), top_features, fontsize=9)
                    plt.xlabel('SHAP Value', fontsize=10, color='white')
                    plt.title('Feature Impact on Prediction', fontsize=12, color='white', pad=15)
                    
                    for i, (bar, val) in enumerate(zip(bars, top_shap_vals)):
                        plt.text(val, i, f' {abs(val):.3f}', 
                                va='center', ha='left' if val > 0 else 'right',
                                fontsize=8, color='white')
                    
                    plt.gcf().set_facecolor("#0f1535")
                    plt.gca().set_facecolor("#13193a")
                    plt.gca().tick_params(axis='x', colors='white', labelsize=9)
                    plt.gca().tick_params(axis='y', colors='white', labelsize=9)
                    plt.gca().spines['bottom'].set_color('#120a8f')
                    plt.gca().spines['left'].set_color('#120a8f')
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().axvline(x=0, color='white', linewidth=0.8, alpha=0.3)
                    
                    plt.subplots_adjust(left=0.4, right=0.95, top=0.95, bottom=0.1)
                    plt.tight_layout()
                    
                    st.pyplot(fig_single)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please ensure your CSV has correct columns and format.")

    elif model is None or explainer is None:
        st.warning("Could not load the model.")
    else:
        st.info("Upload a CSV to begin analysis.")

if __name__ == "__main__":
    main()