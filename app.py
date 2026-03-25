import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(page_title="EV Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")

# === Sidebar & Theme Configuration ===
with st.sidebar:
    st.image("ev-car-factory.jpg", use_container_width=True)
    
    # New Sliding / Segmented Control for Navigation
    st.markdown("### 🧭 Navigation")
    page_sel = st.segmented_control(
        "Navigation", 
        ["Dashboard", "About Project"], 
        default="Dashboard", 
        label_visibility="collapsed"
    )
    # If the user deselects, it returns None. Default back to Dashboard.
    page = page_sel if page_sel else "Dashboard"
    
    st.markdown("---")
    
    st.markdown("### 🎨 UI Theme")
    theme = st.selectbox("Select Theme", ["Professional Light", "Dark Glass", "Midnight Blue", "Crimson Red"], label_visibility="collapsed")
    st.markdown("---")

# === Theme Variables Mapping ===
if theme == "Professional Light":
    bg_app = "#f8f9fc"
    bg_sidebar = "#ffffff"
    text_main = "#1e293b"
    text_muted = "#64748b"
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    accent = "#3b82f6"
    plot_bg = "#ffffff"
    plot_text = "#475569"
elif theme == "Dark Glass":
    bg_app = "#121212"
    bg_sidebar = "#181818"
    text_main = "#f8f9fa"
    text_muted = "#a1a1aa"
    card_bg = "#1e1e1e"
    card_border = "#333333"
    accent = "#10b981"
    plot_bg = "#1e1e1e"
    plot_text = "#e2e8f0"
elif theme == "Midnight Blue":
    bg_app = "#0f172a"
    bg_sidebar = "#0B1120"
    text_main = "#f1f5f9"
    text_muted = "#94a3b8"
    card_bg = "#1e293b"
    card_border = "#334155"
    accent = "#38bdf8"
    plot_bg = "#1e293b"
    plot_text = "#cbd5e1"
elif theme == "Crimson Red":
    bg_app = "#2c0406"
    bg_sidebar = "#1f0304"
    text_main = "#fef2f2"
    text_muted = "#fca5a5"
    card_bg = "#450a0a"
    card_border = "#7f1d1d"
    accent = "#f87171"
    plot_bg = "#450a0a"
    plot_text = "#fecaca"

# === Improved Typography and Dynamic Styling ===
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        /* App Backgrounds */
        .stApp {{
            background-color: {bg_app} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {card_border} !important;
        }}
        
        /* Typography overriding ONLY labels, markdown, and titles to avoid breaking the Streamlit SelectBox Dropdown lists! */
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown li, label p {{
            color: {text_main} !important; 
        }}
        
        /* Card-like containers for metrics and plots */
        .glass-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            border: 1px solid {card_border};
            margin-bottom: 30px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .glass-card:hover {{
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.08);
        }}
        
        /* Metric Styling */
        .metric-label {{
            font-size: 16px;
            color: {text_muted} !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 42px;
            font-weight: 800;
            color: {text_main} !important;
            margin: 8px 0;
            letter-spacing: -1.5px;
            line-height: 1.1;
        }}
        .metric-delta.positive {{ color: {accent} !important; font-weight: 700; font-size: 18px; }}
        .metric-delta.negative {{ color: #ef4444 !important; font-weight: 700; font-size: 18px; }}
        
        /* Header styling */
        .main-header {{
            font-size: 46px;
            font-weight: 800;
            color: {text_main} !important;
            margin-bottom: 12px;
            padding-bottom: 16px;
            border-bottom: 2px solid {card_border};
            letter-spacing: -1.2px;
        }}
        .sub-header {{
            font-size: 22px;
            color: {text_muted} !important;
            margin-bottom: 36px;
            font-weight: 400;
            line-height: 1.5;
        }}
        
        /* Success alert text color */
        .stSuccess {{
            background-color: {card_bg} !important;
            border: 1px solid {accent} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# === Loaders (Cached for Speed) ===
@st.cache_resource(show_spinner=False)
def load_ml_model():
    return joblib.load('forecasting_ev_model.pkl')

@st.cache_data(show_spinner=False)
def load_and_prep_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

with st.spinner('Loading Application...'):
    model = load_ml_model()
    df = load_and_prep_data()
    county_list = sorted(df['County'].dropna().unique().tolist())

# === Finish Sidebar specific to Dashboard ===
if page == "Dashboard":
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        # Primary County Selection
        county = st.selectbox("🎯 Select Primary County", county_list, index=county_list.index("King") if "King" in county_list else 0)
        
        # Comparison Counties
        st.markdown("### 📉 Compare Trending")
        multi_counties = st.multiselect("Select up to 4 counties to compare", county_list, default=[county], max_selections=4)
        st.markdown("---")

# Global Sidebar Footer
with st.sidebar:
    st.markdown("""
    <div style='font-size: 13px; color: #94a3b8; text-align: center;'>
        Prepared for Practising Machine Learning<br>
        <strong>Divyanshu</strong>
    </div>
    """, unsafe_allow_html=True)


# === Helper function for forecasting ===
@st.cache_data(show_spinner=False)
def generate_forecast(target_county):
    county_df = df[df['County'] == target_county].sort_values("Date")
    county_code = county_df['county_encoded'].iloc[0]
    
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    
    # Pad to prevent IndexErrors on regions with less than 6 historical records
    if len(historical_ev) < 6:
        historical_ev = [0] * (6 - len(historical_ev)) + historical_ev
        
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    
    future_rows = []
    forecast_horizon = 36
    
    # Predict step-by-step
    curr_historical_ev = historical_ev.copy()
    curr_cumulative_ev = cumulative_ev.copy()
    
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        
        lag1, lag2, lag3 = curr_historical_ev[-1], curr_historical_ev[-2], curr_historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        
        recent_cumulative = curr_cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0
        
        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }
        
        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
        
        curr_historical_ev.append(pred)
        if len(curr_historical_ev) > 6: curr_historical_ev.pop(0)
        curr_cumulative_ev.append(curr_cumulative_ev[-1] + pred)
        if len(curr_cumulative_ev) > 6: curr_cumulative_ev.pop(0)
            
    # Combine Data
    hist_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    hist_cum['Source'] = 'Historical'
    hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
    
    fc_df = pd.DataFrame(future_rows)
    fc_df['Source'] = 'Forecast'
    fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
    
    combined = pd.concat([
        hist_cum[['Date', 'Cumulative EV', 'Source']],
        fc_df[['Date', 'Cumulative EV', 'Source']]
    ], ignore_index=True)
    
    return combined


# === PAGE ROUTING ===
if page == "Dashboard":
    
    # === Main Layout ===
    st.markdown("<div class='main-header'>⚡ EV Adoption Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predictive analysis of Electric Vehicle growth across Washington State counties over a 3-year horizon.</div>", unsafe_allow_html=True)

    # Generate forecast for primary county
    primary_data = generate_forecast(county)

    historical_total = primary_data[primary_data['Source'] == 'Historical']['Cumulative EV'].iloc[-1]
    forecasted_total = primary_data['Cumulative EV'].iloc[-1]
    growth_amt = forecasted_total - historical_total
    growth_pct = (growth_amt / historical_total) * 100 if historical_total > 0 else 0

    # === Metrics Row ===
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Current Total EVs ({county})</div>
            <div class="metric-value">{int(historical_total):,}</div>
            <div class="metric-delta">As of Last Record</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">3-Year Forecast Total</div>
            <div class="metric-value">{int(forecasted_total):,}</div>
            <div class="metric-delta positive">+{int(growth_amt):,} New EVs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        trend_class = "positive" if growth_pct >= 0 else "negative"
        trend_sign = "+" if growth_pct >= 0 else ""
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Expected Growth</div>
            <div class="metric-value">{trend_sign}{growth_pct:.1f}%</div>
            <div class="metric-delta {trend_class}">36 Month Horizon</div>
        </div>
        """, unsafe_allow_html=True)


    # === Main Plot ===
    st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0;'>📈 Forecast Trajectory: {county} County</h3>", unsafe_allow_html=True)
    st.write("") # Spacing

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    # Matching plot theme dynamically
    fig.patch.set_facecolor(plot_bg)
    ax.set_facecolor(plot_bg)
    ax.grid(True, linestyle='--', alpha=0.3, color=plot_text)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(plot_text)
    ax.spines['bottom'].set_color(plot_text)
    ax.tick_params(colors=plot_text)

    # Plot Historical
    hist_data = primary_data[primary_data['Source'] == 'Historical']
    ax.plot(hist_data['Date'], hist_data['Cumulative EV'], label='Historical', color=accent, linewidth=3)

    # Plot Forecast
    fc_data = primary_data[primary_data['Source'] == 'Forecast']
    # Connect the lines
    connecting_point = pd.concat([hist_data.iloc[-1:], fc_data])
    ax.plot(connecting_point['Date'], connecting_point['Cumulative EV'], label='Forecast', color='#10b981', linewidth=3, linestyle='--')

    ax.set_ylabel("Total EVs", color=plot_text, fontweight='bold')
    ax.legend(frameon=False, loc='upper left', labelcolor=plot_text)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)


    # === Comparison Section ===
    if multi_counties:
        st.markdown("<div class='sub-header' style='margin-bottom: 16px; margin-top: 32px;'>Regional Comparison Analysis</div>", unsafe_allow_html=True)
        
        comparison_data = []
        
        # Progress bar for better UX during calculation
        if len(multi_counties) > 1:
            prog_bar = st.progress(0)
        
        for idx, cty in enumerate(multi_counties):
            cty_data = generate_forecast(cty)
            cty_data['County'] = cty
            comparison_data.append(cty_data)
            if len(multi_counties) > 1:
                prog_bar.progress((idx + 1) / len(multi_counties))
                
        if len(multi_counties) > 1:
            prog_bar.empty()

        comp_df = pd.concat(comparison_data, ignore_index=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=100)
        
        fig2.patch.set_facecolor(plot_bg)
        ax2.set_facecolor(plot_bg)
        ax2.grid(True, linestyle='--', alpha=0.3, color=plot_text)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(plot_text)
        ax2.spines['bottom'].set_color(plot_text)
        ax2.tick_params(colors=plot_text)

        colors = ['#3b82f6', '#f59e0b', '#8b5cf6']
        for (cty, group), color in zip(comp_df.groupby('County'), colors):
            ax2.plot(group['Date'], group['Cumulative EV'], label=cty, linewidth=2.5, color=color)
            
        ax2.set_ylabel("Total Cumulative EVs", color=plot_text, fontweight='bold')
        ax2.legend(frameon=False, loc='upper left', labelcolor=plot_text)
        
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if len(multi_counties) > 1:
            st.success(f"**Comparison Complete:** Trend analysis of the {len(multi_counties)} selected regions completed successfully.")

elif page == "About Project":
    st.markdown("<div class='main-header'>ℹ️ About This Project</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">🔬 Project Goal & Vision</h3>
        <p style="font-size: 18px; line-height: 1.7;">
        This project aims to forecast the adoption of Electric Vehicles (EVs) across various counties in Washington State over a 3-year horizon. By leveraging historical EV registration data and robust machine learning regression algorithms, we provide highly visual and actionable insights for urban planners, utility companies, and policymakers to prepare for future demands—such as installing new charging grid infrastructure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 30px;'>📊 System Architecture & Pipeline</h3>", unsafe_allow_html=True)
    
    st.image(r"C:\Users\hp\.gemini\antigravity\brain\f436b810-4dc1-4716-9e7b-bb9541d5604d\ml_pipeline_architecture_1774429536690.png", use_container_width=True)

    st.markdown("""
    <div class="glass-card" style="margin-top: 24px;">
        <h3 style="margin-top:0;">🧠 Why is this a good Project?</h3>
        <ul style="font-size: 18px; line-height: 1.9; margin-top: 12px;">
            <li><b>End-to-End Delivery:</b> This covers the complete Machine Learning lifecycle: Data ingestion, preprocessing, complex feature engineering (creating lags, rolling means, and slopes), deploying models using <code>joblib</code>, and finally serving it via an interactive presentation dashboard.</li>
            <li><b>Advanced Time-Series strategy via Regression:</b> Instead of basic models, we demonstrated high-level creativity by formulating a time-series problem as a tabular regression task. By manufacturing latency features (Lag 1, Lag 2) and percentage changes over time, we taught models like Random Forests/Decision Trees to intrinsically understand time trends.</li>
            <li><b>Production-Level UI/UX:</b> Features a sleek, responsive, glass-inspired layout combining the beautiful <code>Inter</code> Font, dynamic responsive themes, and segmented sliding navigations.</li>
            <li><b>Tackles a Real-World Problem:</b> Predicting electric vehicle demands bridges the gap between AI and sustainability. It targets an issue heavily funded by green energy initiatives worldwide, showcasing business intelligence alongside technical skill.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; font-size: 14px; opacity: 0.7;'>Developed by <b>Divyanshu</b> for the AICTE Internship Cycle 2 (S4F)</div>", unsafe_allow_html=True)
