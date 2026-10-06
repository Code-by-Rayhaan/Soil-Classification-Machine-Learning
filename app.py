"""
TerrAgro: Soil Classification Using Machine Learning
Case Study 139 | Precision Pedological & Agronomic Intelligence Platform

Authors: Academic Machine Learning Research Team
Framework: Streamlit, Scikit-Learn, Plotly, Pandas, NumPy
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TerrAgro | Soil Classification System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (MODERN GLASSMORPHISM & VIBRANT NATURAL EARTH AESTHETICS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 40%, #52b788 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 35px rgba(27, 67, 50, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.95;
        max-width: 900px;
        line-height: 1.55;
        color: #d8f3dc;
    }

    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #f8f9fa;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(45, 106, 79, 0.12);
    }

    .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
    }

    .card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1b4332;
        margin-top: 0.2rem;
    }

    /* Prediction Result Showcase */
    .result-box {
        background: linear-gradient(145deg, #f8fdf9 0%, #e8f5e9 100%);
        border: 2px solid #52b788;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(45, 106, 79, 0.15);
        margin: 1.5rem 0;
    }

    .soil-name-highlight {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1b4332;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.5rem 0;
    }

    .soil-badge {
        background-color: #2d6a4f;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    .info-callout {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #15803d;
    }

    .crop-tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        margin: 0.2rem;
        font-size: 0.88rem;
        border: 1px solid #bae6fd;
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1.5rem;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. RESOURCE LOADING & CACHING
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")
METRICS_DIR = os.path.join(BASE_DIR, "model_metrics")

@st.cache_resource
def load_all_artifacts():
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    # Load Models
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "KNN": "knn.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl"
    }
    for name, file_name in model_files.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        if os.path.exists(file_path):
            models[name] = joblib.load(file_path)
            
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    label_encoder = None
    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
        
    return models, scaler, label_encoder, metadata

@st.cache_data
def load_dataset():
    data_path = os.path.join(DATASET_DIR, "soil_classification.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_benchmarks():
    bench_path = os.path.join(MODELS_DIR, "benchmark_comparison.csv")
    if os.path.exists(bench_path):
        return pd.read_csv(bench_path)
    return None

@st.cache_data
def load_feature_importance():
    fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None

models, scaler, label_encoder, metadata = load_all_artifacts()
df_data = load_dataset()
df_benchmark = load_benchmarks()
df_feature_importance = load_feature_importance()

# -----------------------------------------------------------------------------
# 4. SOIL DOMAIN METADATA, CROP ADVISORY & AGRONOMIC GUIDES
# -----------------------------------------------------------------------------
SOIL_INFO = {
    "Alluvial Soil": {
        "icon": "🏞️",
        "description": "Rich, fertile soil formed by river silt deposits. High in nitrogen and potassium with balanced moisture retention.",
        "suitability": "Highly Fertile | Excellent for Cereal & Cash Crops",
        "crops": ["Rice", "Wheat", "Sugarcane", "Cotton", "Jute", "Maize", "Oilseeds"],
        "color": "#2a9d8f",
        "ph_range": "6.5 - 7.8 (Neutral to Slightly Alkaline)",
        "management": "Maintain balanced NPK fertilization. Ensure regular organic mulch application to preserve active microbial biomass."
    },
    "Black Soil": {
        "icon": "🌋",
        "description": "Volcanic/basalt-derived clayey soil (Regur) with high calcium, magnesium, and exceptional water-holding capacity.",
        "suitability": "Heavy Texture | Prime for Cotton & Oilseeds",
        "crops": ["Cotton", "Soybean", "Sorghum", "Wheat", "Groundnut", "Pigeon Pea", "Sunflower"],
        "color": "#264653",
        "ph_range": "7.2 - 8.5 (Mildly Alkaline)",
        "management": "Provide adequate drainage during heavy monsoon. Needs supplementary nitrogen and zinc fertilization."
    },
    "Red Soil": {
        "icon": "🧱",
        "description": "Porous, loamy soil with high iron oxide (ferric) content. Leached with moderate moisture retention and mild acidity.",
        "suitability": "Porous & Well-Drained | Ideal for Pulses & Millets",
        "crops": ["Groundnut", "Ragi (Finger Millet)", "Pulses", "Tobacco", "Potato", "Castor", "Sesame"],
        "color": "#e76f51",
        "ph_range": "5.5 - 6.8 (Slightly Acidic)",
        "management": "Apply agricultural lime if pH < 5.8. Incorporate farmyard manure (FYM) to enhance water retention capacity."
    },
    "Laterite Soil": {
        "icon": "🍂",
        "description": "Deeply weathered soil formed under tropical monsoon leaching. Rich in iron and aluminium, deficient in NPK and humus.",
        "suitability": "Intensively Leached | Best for Plantation Crops",
        "crops": ["Tea", "Coffee", "Rubber", "Cashew", "Coconut", "Arecanut", "Cardamom"],
        "color": "#d62828",
        "ph_range": "4.5 - 5.8 (Strongly Acidic)",
        "management": "Requires liming to correct soil acidity. Frequent split applications of phosphate and potash are recommended."
    },
    "Clayey Soil": {
        "icon": "🏺",
        "description": "Fine-textured soil composed of over 40% clay minerals. Extremely high moisture retention and slow water permeability.",
        "suitability": "Waterlogging-Tolerant | Prime for Wetland Cultivation",
        "crops": ["Wetland Paddy", "Broccoli", "Cabbage", "Cauliflower", "Broad Beans", "Mustard"],
        "color": "#457b9d",
        "ph_range": "6.0 - 7.8 (Neutral)",
        "management": "Incorporate coarse sand and organic compost to enhance soil aeration and prevent compaction and waterlogging."
    },
    "Sandy Loam": {
        "icon": "🏖️",
        "description": "Coarse, well-aerated soil with rapid drainage and low nutrient-holding capacity. Highly workable and easy to cultivate.",
        "suitability": "Fast Drainage | Ideal for Root Vegetables & Melons",
        "crops": ["Watermelon", "Muskmelon", "Carrot", "Radish", "Peanut", "Cucumber", "Sweet Potato"],
        "color": "#e9c46a",
        "ph_range": "5.8 - 7.0 (Slightly Acidic to Neutral)",
        "management": "Requires frequent micro-irrigation (drip) and regular slow-release organic nutrient top-dressings."
    },
    "Saline Soil": {
        "icon": "🧂",
        "description": "Soil characterized by high soluble salt accumulation (EC > 2.0 dS/m) causing severe osmotic stress to regular plants.",
        "suitability": "High Salinity | Suitable only for Halophytes & Salt-Tolerant Crops",
        "crops": ["Barley", "Date Palm", "Sugar Beet", "Saltbush", "Bermuda Grass", "Cotton (Saline Tolerant)"],
        "color": "#9d4edd",
        "ph_range": "7.8 - 9.5 (Alkaline/Sodic)",
        "management": "Apply Gypsum ($CaSO_4$) followed by deep leaching with clean water to flush excess sodium from root zone."
    },
    "Peaty Soil": {
        "icon": "🌱",
        "description": "Dark, sponge-like soil enriched with over 5% organic matter and humus. High moisture content with strong natural acidity.",
        "suitability": "Organic Rich | Acidic Wetland Specialties",
        "crops": ["Rice", "Jute", "Spices", "Potatoes", "Blueberries", "Cranberries", "Leafy Vegetables"],
        "color": "#1b4332",
        "ph_range": "3.8 - 5.2 (Strongly Acidic)",
        "management": "Improve sub-surface drainage and apply lime/dolomite to buffer high organic acidity."
    }
}

# -----------------------------------------------------------------------------
# 5. SIDEBAR: MODEL CONTROLS, QUICK PRESETS & METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine & Parameters")
    
    # Model Selection
    selected_model_name = st.selectbox(
        "Select Classification Algorithm:",
        list(models.keys()),
        index=0,
        help="Switch dynamically between all 5 Machine Learning algorithms evaluated in Case Study 139."
    )
    
    current_model = models.get(selected_model_name)
    
    # Show active model accuracy badge
    if df_benchmark is not None:
        row_match = df_benchmark[df_benchmark["Algorithm"] == selected_model_name]
        if not row_match.empty:
            acc_val = row_match.iloc[0]["Accuracy"]
            f1_val = row_match.iloc[0]["F1-Score (Macro)"]
            st.success(f"**{selected_model_name}**  \n🎯 Accuracy: **{acc_val}%** | 🏆 F1: **{f1_val}%**")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Agronomic Presets")
    st.caption("Load verified field measurements from distinct agro-climatic zones:")
    
    preset_choice = st.selectbox(
        "Choose Field Scenario Preset:",
        [
            "Custom Field Input",
            "Indo-Gangetic Plain (Alluvial Loam)",
            "Deccan Plateau (Black Cotton Regur)",
            "Chota Nagpur Plateau (Red Sandy Soil)",
            "Western Ghats Slopes (Laterite Leached)",
            "Lower Bengal Delta (Heavy Clayey Soil)",
            "Rajasthan Arid Zone (Sandy Loam)",
            "Rann of Kutch (High Saline Soil)",
            "Kerala Kuttanad Wetlands (Peaty Organic)"
        ]
    )
    
    # Presets definition: (pH, N, P, K, Moisture, Organic_Matter, EC, Temperature, Humidity)
    presets = {
        "Custom Field Input": (7.0, 150.0, 50.0, 180.0, 30.0, 2.5, 0.8, 27.0, 65.0),
        "Indo-Gangetic Plain (Alluvial Loam)": (7.1, 185.0, 68.0, 200.0, 33.0, 2.85, 0.72, 26.0, 70.0),
        "Deccan Plateau (Black Cotton Regur)": (7.9, 92.0, 40.0, 245.0, 48.0, 1.85, 0.98, 28.5, 52.0),
        "Chota Nagpur Plateau (Red Sandy Soil)": (6.1, 68.0, 26.0, 105.0, 20.0, 1.05, 0.42, 30.0, 45.0),
        "Western Ghats Slopes (Laterite Leached)": (5.0, 48.0, 17.0, 72.0, 17.5, 0.80, 0.28, 31.5, 79.0),
        "Lower Bengal Delta (Heavy Clayey Soil)": (6.8, 145.0, 55.0, 175.0, 54.0, 3.50, 1.20, 24.5, 76.0),
        "Rajasthan Arid Zone (Sandy Loam)": (6.5, 58.0, 22.0, 80.0, 12.5, 0.60, 0.25, 32.0, 38.0),
        "Rann of Kutch (High Saline Soil)": (8.7, 75.0, 32.0, 140.0, 15.0, 0.70, 4.50, 33.5, 35.0),
        "Kerala Kuttanad Wetlands (Peaty Organic)": (4.3, 235.0, 28.0, 88.0, 60.0, 7.20, 1.65, 21.0, 86.0)
    }
    
    def_ph, def_n, def_p, def_k, def_moist, def_om, def_ec, def_temp, def_hum = presets[preset_choice]

    st.markdown("---")
    st.markdown("#### 🧪 Soil Physical & Chemical Inputs")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        inp_ph = st.number_input("pH Level (3.0 - 10.0)", min_value=3.0, max_value=10.0, value=float(def_ph), step=0.1)
        inp_n = st.number_input("Nitrogen (N) kg/ha", min_value=5.0, max_value=350.0, value=float(def_n), step=1.0)
        inp_p = st.number_input("Phosphorus (P) kg/ha", min_value=5.0, max_value=200.0, value=float(def_p), step=1.0)
        inp_k = st.number_input("Potassium (K) kg/ha", min_value=10.0, max_value=350.0, value=float(def_k), step=1.0)
        inp_ec = st.number_input("EC (dS/m)", min_value=0.05, max_value=8.0, value=float(def_ec), step=0.05, help="Electrical Conductivity (Salinity indicator)")
        
    with col_s2:
        inp_moist = st.number_input("Moisture (%)", min_value=5.0, max_value=75.0, value=float(def_moist), step=0.5)
        inp_om = st.number_input("Organic Matter (%)", min_value=0.1, max_value=12.0, value=float(def_om), step=0.1)
        inp_temp = st.number_input("Temperature (°C)", min_value=8.0, max_value=48.0, value=float(def_temp), step=0.5)
        inp_hum = st.number_input("Humidity (%)", min_value=10.0, max_value=98.0, value=float(def_hum), step=1.0)

    st.markdown("---")
    st.markdown("""
    **Project Information:**
    - **Case Study**: CS-139 (Soil Classification)
    - **Dataset**: 2,800 Multi-Parameter Observations
    - **Classes**: 8 Distinct Soil Categories
    - **Evaluation**: 5-Fold Stratified Cross-Validation
    """)

# -----------------------------------------------------------------------------
# 6. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌱 TerrAgro: Soil Classification Intelligence</div>
    <div class="hero-subtitle">
        Automated pedological classification and precision agronomic recommendation platform using 5 Machine Learning algorithms, 
        evaluating 9 fundamental physical, chemical, and environmental soil indicators.
    </div>
    <div style="margin-top: 1.2rem;">
        <span class="badge-pill">🎓 Case Study 139</span>
        <span class="badge-pill">⚡ 5 ML Algorithms (LR, KNN, DT, RF, GB)</span>
        <span class="badge-pill">🏆 Top Model F1: 99.29%</span>
        <span class="badge-pill">🧪 8 Agronomic Soil Classes</span>
        <span class="badge-pill">🔄 5-Fold Stratified CV</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN INTERFACE TABS
# -----------------------------------------------------------------------------
tab_live, tab_batch, tab_benchmarks, tab_eda, tab_casestudy = st.tabs([
    "🧪 Live Soil Classifier",
    "📁 Batch CSV Assessment",
    "📊 Model Benchmark & Comparison",
    "🔬 Exploratory Data Analysis",
    "📚 Case Study & Viva Voce Q&A"
])

# =============================================================================
# TAB 1: LIVE SOIL CLASSIFIER & AGRONOMIC ADVISORY
# =============================================================================
with tab_live:
    st.markdown("### 🧪 Real-Time Soil Classification & Suitability Analysis")
    st.caption("Input soil measurements in the sidebar and trigger instant multi-algorithm prediction.")

    # Form feature array
    input_features = np.array([[inp_ph, inp_n, inp_p, inp_k, inp_moist, inp_om, inp_ec, inp_temp, inp_hum]])
    
    # Scale if selected model requires scaling
    scaling_required = selected_model_name in ["Logistic Regression", "KNN"]
    model_input = scaler.transform(input_features) if (scaling_required and scaler is not None) else input_features
    
    if current_model is not None and label_encoder is not None:
        prediction_idx = current_model.predict(model_input)[0]
        predicted_soil = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Probabilities
        if hasattr(current_model, "predict_proba"):
            probabilities = current_model.predict_proba(model_input)[0]
            confidence = probabilities[prediction_idx] * 100
        else:
            probabilities = [0.0] * len(label_encoder.classes_)
            probabilities[prediction_idx] = 1.0
            confidence = 100.0

        soil_meta = SOIL_INFO.get(predicted_soil, {
            "icon": "🌱", "description": "Classified soil sample.",
            "suitability": "General Agriculture", "crops": ["Wheat", "Rice"],
            "color": "#2d6a4f", "ph_range": "6.0 - 7.5", "management": "Maintain balanced NPK."
        })

        # Display Top Result Showcase
        st.markdown(f"""
        <div class="result-box">
            <span style="font-size: 3.5rem;">{soil_meta['icon']}</span>
            <div style="color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem; margin-top: 0.5rem;">
                Classification Output ({selected_model_name})
            </div>
            <div class="soil-name-highlight">{predicted_soil}</div>
            <div class="soil-badge">Model Confidence: {confidence:.2f}%</div>
            <p style="color: #334155; font-size: 1.05rem; max-width: 750px; margin: 1rem auto 0 auto; line-height: 1.5;">
                {soil_meta['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 4 Key Agronomic Metric Indicators
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # pH Status
        ph_status = "Acidic" if inp_ph < 6.0 else ("Alkaline" if inp_ph > 7.8 else "Optimal Neutral")
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Soil pH Status</div>
                <div class="card-value">{inp_ph:.2f}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">Class: <b>{ph_status}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        # Salinity Status
        sal_status = "High Salinity (Risk)" if inp_ec > 2.0 else ("Moderate" if inp_ec > 1.0 else "Non-Saline (Safe)")
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Electrical Conductivity</div>
                <div class="card-value">{inp_ec:.2f} <span style="font-size: 1rem;">dS/m</span></div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">Status: <b>{sal_status}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Organic Matter
        om_status = "High / Rich" if inp_om > 3.0 else ("Moderate" if inp_om > 1.5 else "Low / Deficient")
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Organic Carbon / Matter</div>
                <div class="card-value">{inp_om:.2f}%</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">Rating: <b>{om_status}</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Moisture Retention
        moist_status = "Saturated / Waterlogged" if inp_moist > 45.0 else ("Well-Hydrated" if inp_moist > 22.0 else "Dry / Low")
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Soil Moisture Content</div>
                <div class="card-value">{inp_moist:.1f}%</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">Hydration: <b>{moist_status}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Two Column Deep-Dive: Agronomic Advice + Probability Distribution
        col_rec, col_prob = st.columns([1.1, 0.9])
        
        with col_rec:
            st.markdown("#### 🌾 Precision Agricultural Suitability & Crop Advisory")
            st.markdown(f"**Agricultural Suitability Index:** `{soil_meta['suitability']}`")
            st.markdown(f"**Standard Typical pH Range:** `{soil_meta['ph_range']}`")
            
            st.markdown("**Top Recommended Crops for this Soil Type:**")
            crop_html = "".join([f"<span class='crop-tag'>🌱 {c}</span>" for c in soil_meta['crops']])
            st.markdown(f"<div style='margin-bottom: 1rem;'>{crop_html}</div>", unsafe_allow_html=True)
            
            st.markdown("💡 **Pedological Management & Fertilizer Prescription:**")
            st.info(soil_meta['management'])
            
            st.markdown("📋 **N-P-K Nutrient Balance Ratio:**")
            total_npk = inp_n + inp_p + inp_k
            if total_npk > 0:
                st.write(f"- Nitrogen (N): **{inp_n/total_npk*100:.1f}%** | Phosphorus (P): **{inp_p/total_npk*100:.1f}%** | Potassium (K): **{inp_k/total_npk*100:.1f}%**")

        with col_prob:
            st.markdown("#### 📈 Model Probability Distribution Across All Classes")
            prob_df = pd.DataFrame({
                "Soil Type": label_encoder.classes_,
                "Probability (%)": [p * 100 for p in probabilities]
            }).sort_values(by="Probability (%)", ascending=True)

            fig_prob = px.bar(
                prob_df, x="Probability (%)", y="Soil Type",
                orientation='h',
                color="Probability (%)",
                color_continuous_scale="Viridis",
                text=prob_df["Probability (%)"].apply(lambda v: f"{v:.1f}%")
            )
            fig_prob.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Confidence Probability (%)",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)

# =============================================================================
# TAB 2: BATCH CSV ASSESSMENT (AUTOMATED MULTI-SAMPLE CLASSIFICATION)
# =============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Soil Sample Classification via CSV Upload")
    st.write("Upload an agronomic CSV file containing soil test samples to classify multiple field observations simultaneously.")

    # Download Template Button
    sample_csv_path = os.path.join(DATASET_DIR, "test_samples.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample Test CSV Template (25 Unlabeled Samples)",
                data=f,
                file_name="soil_test_sample_template.csv",
                mime="text/csv",
                help="Click to download a formatted test CSV template ready for instant upload."
            )

    uploaded_file = st.file_uploader("Upload Soil Testing Dataset (CSV):", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✔️ Successfully loaded CSV with {len(batch_df)} samples.")
            
            required_cols = ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", 
                             "Organic_Matter", "Electrical_Conductivity", "Temperature", "Humidity"]
            
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns in CSV: {missing_cols}")
                st.info(f"Required column names: {required_cols}")
            else:
                X_batch = batch_df[required_cols].copy()
                
                # Perform inference
                if selected_model_name in ["Logistic Regression", "KNN"] and scaler is not None:
                    X_infer = scaler.transform(X_batch)
                else:
                    X_infer = X_batch.values
                    
                preds_idx = current_model.predict(X_infer)
                preds_labels = label_encoder.inverse_transform(preds_idx)
                
                batch_df["Predicted_Soil_Type"] = preds_labels
                
                if hasattr(current_model, "predict_proba"):
                    probs = current_model.predict_proba(X_infer)
                    batch_df["Confidence (%)"] = [round(np.max(p) * 100, 2) for p in probs]

                # Map recommended crops
                batch_df["Recommended_Crops"] = batch_df["Predicted_Soil_Type"].apply(
                    lambda x: ", ".join(SOIL_INFO.get(x, {}).get("crops", ["Cereals"])[:4])
                )
                
                st.markdown("#### 📋 Classified Soil Assessment Results")
                st.dataframe(batch_df, use_container_width=True)
                
                # Visual summary of batch results
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.markdown("##### 🥧 Soil Type Distribution in Batch")
                    fig_pie = px.pie(
                        batch_df, names="Predicted_Soil_Type",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        hole=0.4
                    )
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_b2:
                    st.markdown("##### 📊 Average Confidence by Soil Type")
                    if "Confidence (%)" in batch_df.columns:
                        conf_summary = batch_df.groupby("Predicted_Soil_Type")["Confidence (%)"].mean().reset_index()
                        fig_conf = px.bar(
                            conf_summary, x="Predicted_Soil_Type", y="Confidence (%)",
                            color="Confidence (%)", color_continuous_scale="Teal"
                        )
                        fig_conf.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
                        st.plotly_chart(fig_conf, use_container_width=True)

                # Download Classified CSV
                csv_export = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Batch Soil Classification Report (CSV)",
                    data=csv_export,
                    file_name="classified_soil_report.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

# =============================================================================
# TAB 3: MODEL BENCHMARK & COMPARATIVE STUDY (CASE STUDY OBJECTIVES 4 & 5)
# =============================================================================
with tab_benchmarks:
    st.markdown("### 📊 Comprehensive Model Benchmarking & Algorithmic Comparison")
    st.write("Rigorous quantitative comparative study of all 5 Machine Learning classifiers evaluated under Stratified 5-Fold Cross Validation.")

    if df_benchmark is not None:
        st.markdown("#### 🏆 Multi-Metric Algorithmic Leaderboard")
        st.dataframe(
            df_benchmark.style.highlight_max(subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "F1-Score (Weighted)"], color="#bbf7d0"),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### 📈 Accuracy & Macro F1-Score Comparison")
            fig_bar = px.bar(
                df_benchmark, x="Algorithm", y=["Accuracy", "F1-Score (Macro)"],
                barmode='group',
                color_discrete_sequence=['#2a9d8f', '#e76f51']
            )
            fig_bar.update_layout(yaxis_range=[85, 102], height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            st.markdown("#### 🔬 Impact of StandardScaler Preprocessing (Case Study Q5)")
            prep_path = os.path.join(MODELS_DIR, "preprocessing_impact.csv")
            if os.path.exists(prep_path):
                df_prep = pd.read_csv(prep_path)
                fig_prep = px.bar(
                    df_prep, x="Algorithm", y=["Without Scaling (Raw)", "With StandardScaler"],
                    barmode='group',
                    color_discrete_sequence=['#e63946', '#457b9d']
                )
                fig_prep.update_layout(yaxis_range=[80, 102], height=350, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_prep, use_container_width=True)
                st.caption("Key Finding: Distance-based KNN and Gradient-based Logistic Regression experience dramatic gains from StandardScaler, while Tree ensembles remain invariant.")

        # Interactive Confusion Matrix Visualizer
        st.markdown("---")
        st.markdown("#### 🔍 Interactive Multi-Class Confusion Matrix Visualizer")
        cm_model_choice = st.selectbox("Select Model to Inspect Confusion Matrix:", list(models.keys()), index=0)
        
        if metadata and "confusion_matrices" in metadata:
            cm_data = np.array(metadata["confusion_matrices"][cm_model_choice])
            target_classes = metadata["target_classes"]
            
            fig_cm = px.imshow(
                cm_data,
                labels=dict(x="Predicted Soil Type", y="Actual Soil Type", color="Sample Count"),
                x=target_classes,
                y=target_classes,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto"
            )
            fig_cm.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance Analysis
        if df_feature_importance is not None:
            st.markdown("---")
            st.markdown("#### 🌟 Soil Characteristic Importance Ranking (Case Study Q2)")
            fig_fi = px.bar(
                df_feature_importance, x="Average Importance", y="Feature",
                orientation='h',
                color="Average Importance",
                color_continuous_scale="tealgrn",
                text=df_feature_importance["Average Importance"].apply(lambda v: f"{v*100:.1f}%")
            )
            fig_fi.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)

# =============================================================================
# TAB 4: EXPLORATORY DATA ANALYSIS & SOIL DYNAMICS
# =============================================================================
with tab_eda:
    st.markdown("### 🔬 Exploratory Data Analysis & Pedological Distributions")
    st.write("Interactive exploration of physical and chemical soil properties across all 8 agronomic classes.")

    if df_data is not None:
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown("#### 🌡️ Parameter Distribution by Soil Class")
            eda_feature = st.selectbox(
                "Select Soil Characteristic to Inspect:",
                ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture", "Organic_Matter", "Electrical_Conductivity", "Temperature", "Humidity"],
                index=0
            )
            fig_box = px.box(
                df_data, x="Soil_Type", y=eda_feature, color="Soil_Type",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_box, use_container_width=True)

        with col_e2:
            st.markdown("#### 🧬 Acidity (pH) vs. Salinity (EC) Clustering")
            fig_scatter = px.scatter(
                df_data, x="pH", y="Electrical_Conductivity", color="Soil_Type",
                size="Moisture", hover_data=["Organic_Matter", "Nitrogen"],
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scatter.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Salinity Threshold (2.0 dS/m)")
            fig_scatter.add_vline(x=7.0, line_dash="dot", line_color="gray", annotation_text="Neutral pH (7.0)")
            fig_scatter.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        col_e3, col_e4 = st.columns(2)
        
        with col_e3:
            st.markdown("#### 💧 Moisture (%) vs. Organic Matter (%) Relationship")
            fig_moist = px.scatter(
                df_data, x="Organic_Matter", y="Moisture", color="Soil_Type",
                hover_data=["pH", "Nitrogen", "Potassium"],
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_moist.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_moist, use_container_width=True)

        with col_e4:
            st.markdown("#### 🌾 Macronutrient (NPK) Distribution Profile")
            fig_npk = px.scatter_3d(
                df_data, x='Nitrogen', y='Phosphorus', z='Potassium',
                color='Soil_Type', opacity=0.7, size_max=10
            )
            fig_npk.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_npk, use_container_width=True)

# =============================================================================
# TAB 5: CASE STUDY ANSWERS & VIVA VOCE PREPARATION HUB
# =============================================================================
with tab_casestudy:
    st.markdown("### 📚 Case Study 139: Final Analysis & Core Questions Answered")
    st.write("Rigorous, data-backed technical answers to the 6 core questions formulated in Case Study 139.")

    st.markdown("""
    <div class="info-callout">
        <b>Case Study Objective Validation:</b> All findings below are derived from empirical cross-validation runs on 2,800 balanced soil observations across 8 distinct categories.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("❓ Question 1: Can soil type be automatically classified?", expanded=True):
        st.markdown("""
        **Answer: Yes, with exceptionally high accuracy (> 99%).**  
        Soil types possess distinct physicochemical signatures across pH, macronutrients (N-P-K), organic carbon, moisture retention, and electrical conductivity. 
        Machine learning models successfully learn the multi-dimensional decision boundaries separating these classes. 
        Both **Logistic Regression (99.29%)** and **Random Forest (99.11%)** achieve near-perfect discrimination under 5-Fold Stratified Cross-Validation.
        """)

    with st.expander("❓ Question 2: Which soil parameter contributes most to classification?", expanded=True):
        st.markdown("""
        **Answer: Electrical Conductivity (EC - 18.84%) and Organic Matter (17.56%).**  
        - **Electrical Conductivity (EC)** is the sharpest differentiator because saline and sodic soils exhibit extreme EC values (> 2.5–6.0 dS/m) compared to non-saline soils (< 1.0 dS/m).
        - **Organic Matter (%)** provides a distinct fingerprint for Peaty/Bog soils (6.0–9.0%) versus mineral-poor sandy and laterite soils (< 1.0%).
        - **Moisture (%)** (16.10%) and **pH** (10.14%) follow closely, effectively distinguishing heavy waterlogging clays from well-drained sandy loams and acidic laterites.
        """)

    with st.expander("❓ Question 3: Which algorithm provides the highest F1-score?", expanded=True):
        st.markdown("""
        **Answer: Logistic Regression (Macro F1: 99.29%) and Random Forest (Macro F1: 99.11%).**  
        - **Logistic Regression** with `StandardScaler` provides outstanding linear separability when normalized across standard z-scores, reaching a 5-Fold CV score of **98.54% ± 0.86%**.
        - **Random Forest** provides nearly identical performance (**99.11% Test F1, 98.50% CV**) with the added advantage of being completely invariant to unscaled feature magnitudes and robust against outliers.
        """)

    with st.expander("❓ Question 4: Which soil categories are frequently confused?", expanded=True):
        st.markdown("""
        **Answer: Sandy Loam vs. Red Soil (and Alluvial Soil vs. Clayey Soil).**  
        - **Sandy Loam and Red Soil** exhibit slight boundary overlap because both feature low moisture retention (12–22%), low organic carbon (0.5–1.2%), and mild acidity.
        - **Alluvial and Clayey Soils** share overlapping neutral pH (6.8–7.2) and high potassium levels, differing primarily in moisture retention and clay fraction.
        - In contrast, extreme categories (Saline Soil and Peaty Soil) achieve **100% precision and recall with 0 misclassifications**.
        """)

    with st.expander("❓ Question 5: Does preprocessing improve classification?", expanded=True):
        st.markdown("""
        **Answer: Yes, significantly for distance-based and gradient-based models.**  
        - **K-Nearest Neighbors (KNN):** Accuracy improved from **92.50% (Raw)** to **98.04% (Scaled)** — a massive **+5.54% gain**. Without scaling, features with large numerical ranges (e.g. Potassium: 250 kg/ha) dominate distance metrics over small-range features (e.g. Electrical Conductivity: 0.8 dS/m).
        - **Logistic Regression:** Without scaling, the gradient descent solver failed to converge within 1000 iterations. Applying `StandardScaler` stabilized gradients and boosted accuracy to **99.29%**.
        - **Tree Ensembles (DT, RF, GB):** Invariant to monotonic scaling (0% change in split criteria).
        """)

    with st.expander("❓ Question 6: Can the model classify a completely new soil sample?", expanded=True):
        st.markdown("""
        **Answer: Yes, seamlessly through the deployment pipeline.**  
        The serialized pipeline (`scaler.pkl` + `logistic_regression.pkl` / `random_forest.pkl`) takes raw arbitrary field measurements, scales them to the training distribution, computes probabilistic class likelihoods, and outputs the classified soil category along with confidence percentages and tailored agricultural recommendations.
        """)

    st.markdown("---")
    st.markdown("### 🎓 Viva-Voce Examination Preparation Guide (CS401 / ML202)")
    
    with st.expander("📖 View Top 10 Technical Viva Questions & Examiner Answers"):
        st.markdown(r"""
        1. **Q: Why is Stratified K-Fold CV preferred over regular K-Fold?**  
           *A: Stratified K-Fold preserves the exact percentage of each soil class in every fold, preventing class distribution bias.*
           
        2. **Q: How does Random Forest calculate Feature Importance?**  
           *A: Via Mean Decrease in Impurity (Gini Importance) — calculating the total reduction in Gini impurity brought by that feature across all decision trees in the forest.*
           
        3. **Q: Why does KNN fail without feature scaling?**  
           *A: KNN computes Euclidean distance $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$. A feature with range 0–300 (Potassium) mathematically overwhelms a feature with range 0–2 (EC), distorting neighborhood calculation.*
           
        4. **Q: What is the difference between Macro F1-Score and Weighted F1-Score?**  
           *A: Macro F1 calculates the arithmetic mean of F1 scores across all classes giving equal weight to each class. Weighted F1 weights each class score by its support (sample count).*
           
        5. **Q: How does Gradient Boosting differ from Random Forest?**  
           *A: Random Forest builds independent trees in parallel (Bagging) and averages outputs. Gradient Boosting builds trees sequentially (Boosting), where each tree corrects the residual errors of prior trees.*
        """)

# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.88rem; padding: 1rem 0;">
    🌱 <b>TerrAgro Precision Soil Classification Platform</b> | Case Study 139 Evaluation | Machine Learning Laboratory Project 2026
</div>
""", unsafe_allow_html=True)
