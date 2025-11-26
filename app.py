import io
import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Iris SVM Classifier — Professional ML Studio",
    layout="wide",
    page_icon="🌸",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Modern, Professional Design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    background-attachment: fixed;
}

/* Block Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Hero Section */
.hero-section {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.05), 0 0 0 1px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.3rem;
    color: #374151;
    font-weight: 500;
    margin-bottom: 2rem;
}

/* Stats Cards */
.stats-container {
    display: flex;
    gap: 1.5rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}

.stat-card {
    flex: 1;
    min-width: 200px;
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #4f46e5;
    margin-bottom: 0.25rem;
}

.stat-label {
    font-size: 0.95rem;
    color: #4b5563;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.glass-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    padding: 0.5rem;
    border-radius: 16px;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    color: #4b5563;
    border: 1px solid transparent;
    transition: all 0.25s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.8);
    color: #4f46e5;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: white;
    color: #4f46e5;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid rgba(0, 0, 0, 0.05);
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

/* Metric Cards */
.metric-card {
    background: #f0fdf4;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #bbf7d0;
    text-align: center;
}

.metric-value {
    font-size: 3rem;
    font-weight: 800;
    color: #15803d;
}

.metric-label {
    font-size: 1rem;
    color: #374151;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Prediction Result */
.prediction-result {
    background: #fdf2f8;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    border: 2px solid #fbcfe8;
    margin: 1.5rem 0;
}

.prediction-species {
    font-size: 3rem;
    font-weight: 800;
    color: #db2777;
    margin-bottom: 0.5rem;
}

.prediction-confidence {
    font-size: 1.5rem;
    color: #374151;
    font-weight: 600;
}

/* Feature Cards */
.feature-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.feature-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

/* Buttons */
.stButton > button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2), 0 2px 4px -1px rgba(79, 70, 229, 0.1);
}

.stButton > button:hover {
    background: #4338ca;
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3), 0 4px 6px -2px rgba(79, 70, 229, 0.15);
}

/* Download Button */
.stDownloadButton > button {
    background: #10b981;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2), 0 2px 4px -1px rgba(16, 185, 129, 0.1);
}

.stDownloadButton > button:hover {
    background: #059669;
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3), 0 4px 6px -2px rgba(16, 185, 129, 0.15);
}

/* Sliders */
[data-testid="stSlider"] > div {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

/* DataFrames */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.05);
}

/* Info Boxes */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

.success-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Text Colors */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

p, label, span, div {
    color: #374151 !important;
}

.stMarkdown {
    color: #374151;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    
    .stats-container {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🌸 Iris SVM Classifier Studio</div>
    <div class="hero-subtitle">
        Advanced Support Vector Machine classification for the classic Iris dataset.
        Train, visualize, and export ML models with professional-grade analytics.
    </div>
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">100%</div>
            <div class="stat-label">Max Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">3</div>
            <div class="stat-label">Species</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">150</div>
            <div class="stat-label">Samples</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_iris_df():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

def train_model(X, y, C=1.0, kernel='rbf', gamma='scale', probability=True):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
    model.fit(Xs, y)
    return model, scaler

def predict_with_model(model, scaler, X):
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)
    pred = model.predict(Xs)
    return pred, probs

def model_to_bytes(model, scaler):
    bio = io.BytesIO()
    pickle.dump({'model': model, 'scaler': scaler}, bio)
    bio.seek(0)
    return bio.read()

# Load data
df, iris = load_iris_df()

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Model Configuration")

# Feature Selection
use_all_features = st.sidebar.checkbox('Use All 4 Features', value=True, help='Use all 4 features for 100% accuracy, or just 2 features for ~90% accuracy')

if use_all_features:
    st.sidebar.success('✅ Using 4 features (Expected: ~100% accuracy)')
else:
    st.sidebar.warning('⚠️ Using 2 features (Expected: ~90% accuracy)')

st.sidebar.markdown("---")

# Hyperparameters
st.sidebar.markdown("### 🎛️ Hyperparameters")

C = st.sidebar.slider(
    'C (Regularization)',
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01,
    help='Regularization parameter. Higher values = less regularization'
)

kernel = st.sidebar.selectbox(
    'Kernel Function',
    options=['rbf', 'linear', 'poly', 'sigmoid'],
    index=0,
    help='Kernel type for SVM: RBF (default), Linear, Polynomial, or Sigmoid'
)

gamma = st.sidebar.selectbox(
    'Gamma',
    options=['scale', 'auto'],
    index=0,
    help='Kernel coefficient. "scale" uses 1/(n_features * X.var())'
)

st.sidebar.markdown("---")

# Quick Presets
st.sidebar.markdown("### 🚀 Quick Presets")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button('🎯 Optimal', use_container_width=True):
        C, kernel, gamma = 1.0, 'rbf', 'scale'
        st.rerun()

with col2:
    if st.button('📐 Linear', use_container_width=True):
        C, kernel, gamma = 1.0, 'linear', 'scale'
        st.rerun()

retrain = st.sidebar.button('🔄 Retrain Model', use_container_width=True, type='primary')

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**Iris Dataset**: Classic ML dataset with 150 samples of iris flowers across 3 species.

**SVM**: Support Vector Machine finds optimal hyperplanes to separate classes.

**Made with ❤️ by Ihsan Saif**
""")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.use_all_features = use_all_features

# Check if feature selection has changed
feature_selection_changed = ('use_all_features' in st.session_state and 
                             st.session_state.use_all_features != use_all_features)

# Train model if needed
if (not st.session_state.trained) or retrain or feature_selection_changed:
    if use_all_features:
        X = df[iris.feature_names].values
    else:
        X = df[['sepal length (cm)', 'sepal width (cm)']].values
    
    y = df['target'].values
    model, scaler = train_model(X, y, C=C, kernel=kernel, gamma=gamma, probability=True)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.trained = True
    st.session_state.use_all_features = use_all_features

# Tabs
tabs = st.tabs(["🎯 Predict", "📊 Visualize", "📈 Metrics & Performance", "💾 Dataset & Export"])

# ==================== TAB 1: PREDICT ====================
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 Live Prediction Engine")
    st.markdown("Adjust the flower measurements below to predict the iris species in real-time.")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📏 Feature Inputs")
        
        # Sepal Length
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-label">🌿 Sepal Length (cm)</div>', unsafe_allow_html=True)
        s_len = st.slider('', float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[0,0]), key='s_len', label_visibility='collapsed')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sepal Width
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-label">🌿 Sepal Width (cm)</div>', unsafe_allow_html=True)
        s_wid = st.slider('', float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[0,1]), key='s_wid', label_visibility='collapsed')
        st.markdown('</div>', unsafe_allow_html=True)
        
        if use_all_features:
            # Petal Length
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<div class="feature-label">🌺 Petal Length (cm)</div>', unsafe_allow_html=True)
            p_len = st.slider('', float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[0,2]), key='p_len', label_visibility='collapsed')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Petal Width
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<div class="feature-label">🌺 Petal Width (cm)</div>', unsafe_allow_html=True)
            p_wid = st.slider('', float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[0,3]), key='p_wid', label_visibility='collapsed')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            p_len = 1.4
            p_wid = 0.2
        
        # Callback to set sample values
        def set_sample_values(sl, sw, pl, pw):
            st.session_state.s_len = sl
            st.session_state.s_wid = sw
            st.session_state.p_len = pl
            st.session_state.p_wid = pw

        st.markdown("### 🎲 Quick Samples")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.button('� Setosa', on_click=set_sample_values, args=(5.0, 3.6, 1.4, 0.2), use_container_width=True)
        
        with col_b:
            st.button('🌼 Versicolor', on_click=set_sample_values, args=(6.0, 2.7, 4.2, 1.3), use_container_width=True)
        
        with col_c:
            st.button('🌺 Virginica', on_click=set_sample_values, args=(6.5, 3.0, 5.5, 2.0), use_container_width=True)
    
    with col2:
        st.markdown("### 🔮 Prediction Results")
        
        # Make prediction
        if use_all_features:
            Xnew = np.array([[s_len, s_wid, p_len, p_wid]])
        else:
            Xnew = np.array([[s_len, s_wid]])
        
        pred, probs = predict_with_model(st.session_state.model, st.session_state.scaler, Xnew)
        label = iris.target_names[pred[0]]
        confidence = probs[0][pred[0]] * 100
        
        # Prediction Display
        st.markdown(f"""
        <div class="prediction-result">
            <div class="prediction-species">{label.upper()}</div>
            <div class="prediction-confidence">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Distribution
        st.markdown("#### 📊 Probability Distribution")
        prob_df = pd.DataFrame({
            'Species': iris.target_names,
            'Probability': probs[0] * 100
        })
        
        fig = go.Figure()
        colors = ['#db2777', '#7c3aed', '#4f46e5']
        
        for idx, row in prob_df.iterrows():
            fig.add_trace(go.Bar(
                y=[row['Species']],
                x=[row['Probability']],
                orientation='h',
                name=row['Species'],
                marker=dict(color=colors[idx]),
                text=f"{row['Probability']:.1f}%",
                textposition='auto',
            ))
        
        fig.update_layout(
            showlegend=False,
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title='Probability (%)', range=[0, 100]),
            yaxis=dict(title=''),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Probabilities
        st.markdown("#### 📋 Detailed Probabilities")
        st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}), use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: VISUALIZE ====================
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Data Visualization & Decision Boundaries")
    
    # PCA Projection
    st.markdown("### 🎨 PCA 2D Projection")
    st.info("📘 **Principal Component Analysis (PCA)** reduces the 4D feature space to 2D for visualization while preserving maximum variance.")
    
    X = df[iris.feature_names].values
    y = df['target'].values
    
    # Perform PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(StandardScaler().fit_transform(X))
    
    # Train model on PCA data
    pca_model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    pca_model.fit(X2, y)
    
    # Create decision boundary
    xmin, xmax = X2[:,0].min()-1, X2[:,0].max()+1
    ymin, ymax = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pca_model.predict(grid).reshape(xx.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add contour for decision boundary
    fig.add_trace(go.Contour(
        x=np.linspace(xmin, xmax, 200),
        y=np.linspace(ymin, ymax, 200),
        z=Z,
        showscale=False,
        opacity=0.3,
        colorscale=[[0, '#db2777'], [0.5, '#7c3aed'], [1, '#4f46e5']],
        contours=dict(showlines=False)
    ))
    
    # Add scatter points
    colors_map = {0: '#db2777', 1: '#7c3aed', 2: '#4f46e5'}
    for i, species in enumerate(iris.target_names):
        mask = df['target'] == i
        fig.add_trace(go.Scatter(
            x=X2[mask, 0],
            y=X2[mask, 1],
            mode='markers',
            name=species,
            marker=dict(
                size=10,
                color=colors_map[i],
                line=dict(width=1, color='white')
            )
        ))
    
    fig.update_layout(
        title='PCA Scatter Plot with Decision Boundaries',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        height=500,
        hovermode='closest',
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Distributions
    st.markdown("### 📈 Feature Distributions by Species")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot for sepal length
        fig = go.Figure()
        for i, species in enumerate(iris.target_names):
            mask = df['species'] == species
            fig.add_trace(go.Box(
                y=df[mask]['sepal length (cm)'],
                name=species,
                marker_color=colors_map[i]
            ))
        
        fig.update_layout(
            title='Sepal Length Distribution',
            yaxis_title='Length (cm)',
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot for petal length
        fig = go.Figure()
        for i, species in enumerate(iris.target_names):
            mask = df['species'] == species
            fig.add_trace(go.Box(
                y=df[mask]['petal length (cm)'],
                name=species,
                marker_color=colors_map[i]
            ))
        
        fig.update_layout(
            title='Petal Length Distribution',
            yaxis_title='Length (cm)',
            height=350,
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 3: METRICS ====================
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Model Performance & Metrics")
    
    # Prepare data for metrics
    if use_all_features:
        X_metrics = df[iris.feature_names].values
    else:
        X_metrics = df[['sepal length (cm)', 'sepal width (cm)']].values
    
    y_metrics = df['target'].values
    
    # Cross-validation
    st.markdown("### 🎯 Cross-Validation Results")
    scores = cross_val_score(
        SVC(C=C, kernel=kernel, gamma=gamma, probability=True),
        StandardScaler().fit_transform(X_metrics),
        y_metrics,
        cv=5
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{scores.mean():.1%}</div>
            <div class="metric-label">Mean Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(67, 56, 202, 0.1) 100%); border-color: rgba(79, 70, 229, 0.3);">
            <div class="metric-value" style="color: #4f46e5;">{scores.std():.1%}</div>
            <div class="metric-label">Std Deviation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, rgba(219, 39, 119, 0.1) 0%, rgba(190, 24, 93, 0.1) 100%); border-color: rgba(219, 39, 119, 0.3);">
            <div class="metric-value" style="color: #db2777;">{scores.max():.1%}</div>
            <div class="metric-label">Best Fold</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Fold-by-fold results
    st.markdown("#### 📊 Fold-by-Fold Accuracy")
    fold_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(scores))],
        'Accuracy': scores * 100
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fold_df['Fold'],
        y=fold_df['Accuracy'],
        marker=dict(
            color=fold_df['Accuracy'],
            colorscale=[[0, '#db2777'], [0.5, '#7c3aed'], [1, '#15803d']],
            showscale=False
        ),
        text=[f'{acc:.1f}%' for acc in fold_df['Accuracy']],
        textposition='auto',
    ))
    
    fig.update_layout(
        height=300,
        yaxis=dict(title='Accuracy (%)', range=[0, 100]),
        xaxis=dict(title=''),
        plot_bgcolor='rgba(255,255,255,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### 🎭 Confusion Matrix")
    
    # Train-test split for confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X_metrics, y_metrics, test_size=0.2, random_state=42)
    scaler_cm = StandardScaler()
    X_train_scaled = scaler_cm.fit_transform(X_train)
    X_test_scaled = scaler_cm.transform(X_test)
    
    model_cm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    model_cm.fit(X_train_scaled, y_train)
    y_pred = model_cm.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=iris.target_names,
        y=iris.target_names,
        colorscale=[[0, '#f0f9ff'], [0.5, '#7c3aed'], [1, '#4f46e5']],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16, "color": "white"},
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix Heatmap',
        xaxis_title='Predicted Species',
        yaxis_title='Actual Species',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    
    report_df = pd.DataFrame({
        'Species': iris.target_names,
        'Precision': [report[species]['precision'] * 100 for species in iris.target_names],
        'Recall': [report[species]['recall'] * 100 for species in iris.target_names],
        'F1-Score': [report[species]['f1-score'] * 100 for species in iris.target_names],
        'Support': [int(report[species]['support']) for species in iris.target_names]
    })
    
    st.dataframe(
        report_df.style.format({
            'Precision': '{:.1f}%',
            'Recall': '{:.1f}%',
            'F1-Score': '{:.1f}%',
            'Support': '{:.0f}'
        }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True
    )
    
    # Feature Importance (for linear kernel)
    if kernel == 'linear':
        st.markdown("### 🎯 Feature Importance")
        st.info("📘 Feature importance is calculated from the linear SVM coefficients (absolute values averaged across classes).")
        
        lin_model = SVC(C=C, kernel='linear', probability=True)
        lin_model.fit(StandardScaler().fit_transform(X_metrics), y_metrics)
        
        if use_all_features:
            feature_names = iris.feature_names
        else:
            feature_names = ['sepal length (cm)', 'sepal width (cm)']
        
        coefs = np.mean(np.abs(lin_model.coef_), axis=0)
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefs
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=feat_imp['Feature'],
            x=feat_imp['Importance'],
            orientation='h',
            marker=dict(
                color=feat_imp['Importance'],
                colorscale=[[0, '#db2777'], [1, '#4f46e5']],
                showscale=False
            ),
            text=[f'{imp:.3f}' for imp in feat_imp['Importance']],
            textposition='auto',
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title='Importance (Coefficient Magnitude)',
            yaxis_title='',
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 4: DATASET & EXPORT ====================
with tabs[3]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 💾 Dataset Explorer & Model Export")
    
    # Dataset Statistics
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(iris.feature_names)}</div>
            <div class="stat-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(iris.target_names)}</div>
            <div class="stat-label">Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">0</div>
            <div class="stat-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Class Distribution
    st.markdown("### 🌸 Class Distribution")
    
    class_counts = df['species'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.4,
        marker=dict(colors=['#db2777', '#7c3aed', '#4f46e5']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        height=350,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Statistics
    st.markdown("### 📈 Feature Statistics")
    
    stats_df = df[iris.feature_names].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Max']
    
    st.dataframe(
        stats_df.style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )
    
    # Sample Data
    st.markdown("### 🔍 Sample Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**First 10 rows of the dataset:**")
    
    with col2:
        show_all = st.checkbox('Show all rows', value=False)
    
    if show_all:
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.dataframe(df.head(10), use_container_width=True)
    
    # Export Section
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Download Trained Model")
        st.markdown("Export the trained SVM model with scaler for deployment.")
        
        model_bytes = model_to_bytes(st.session_state.model, st.session_state.scaler)
        
        st.download_button(
            label='📦 Download Model (.pkl)',
            data=model_bytes,
            file_name=f'iris_svm_{kernel}_C{C}.pkl',
            mime='application/octet-stream',
            use_container_width=True
        )
        
        if st.button('💾 Save to Workspace', use_container_width=True):
            out_path = Path(f'iris_svm_{kernel}_C{C}.pkl')
            out_path.write_bytes(model_bytes)
            st.success(f'✅ Model saved to: `{out_path.resolve()}`')
    
    with col2:
        st.markdown("#### 📊 Download Dataset")
        st.markdown("Export the Iris dataset in CSV format.")
        
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label='📄 Download Dataset (.csv)',
            data=csv,
            file_name='iris_dataset.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    # Model Information
    st.markdown("### ℹ️ Current Model Information")
    
    model_info = f"""
    | Parameter | Value |
    |-----------|-------|
    | **Kernel** | {kernel.upper()} |
    | **C (Regularization)** | {C} |
    | **Gamma** | {gamma} |
    | **Features Used** | {'All 4 features' if use_all_features else '2 features (sepal only)'} |
    | **Probability Estimates** | Enabled |
    | **Training Samples** | {len(df)} |
    """
    
    st.markdown(model_info)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.6); border-radius: 16px; margin-top: 2rem;'>
    <h3 style='margin-bottom: 0.5rem;'>🌸 Iris SVM Classifier Studio</h3>
    <p style='color: #6b7280; margin-bottom: 1rem;'>
        Built with Streamlit, Scikit-learn, and Plotly
    </p>
    <p style='color: #4f46e5; font-weight: 600;'>
        Made with ❤️ by <strong>Ihsan Saif</strong> for <strong>Sir Zeeshan</strong>
    </p>
</div>
""", unsafe_allow_html=True)