import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Behind the Diagnosis: Alzheimer’s by Data",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; color: #2ca02c;'>\U0001F4CA Behind the Diagnosis: Alzheimer’s by Data</h1>
    <h4 style='text-align: center;'>A Global View into Patterns, Risk Factors, and Demographics</h4>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    df = pd.read_csv("alzheimers_prediction_dataset.csv")
    df["Diagnosis_Binary"] = df["Alzheimer’s Diagnosis"].map({"Yes": 1, "No": 0})
    df["Age Group"] = pd.cut(df["Age"], bins=[49, 59, 69, 79, 89, 99],
                             labels=["50–59", "60–69", "70–79", "80–89", "90–99"])
    return df

df = load_data()

# --- Section: Diagnosis, Age Group & Correlation --- #
with st.expander("📊 Diagnosis, Age Group & Correlation", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.histogram(df, x="Alzheimer’s Diagnosis", color="Alzheimer’s Diagnosis",
                           color_discrete_map={"Yes": "#2ca02c", "No": "#7f7f7f"})
        fig1.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="Age Group", color="Alzheimer’s Diagnosis", barmode="group",
                           color_discrete_map={"Yes": "#2ca02c", "No": "#7f7f7f"})
        fig2.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        fig_corr, ax = plt.subplots(figsize=(5, 4))
        num_features = ["Age", "BMI", "Education Level", "Cognitive Test Score", "Diagnosis_Binary"]
        corr_matrix = df[num_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="Greens", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig_corr, use_container_width=True)

# --- Section: Cognitive Score Analysis --- #
with st.expander("🧠 Cognitive Score Analysis", expanded=False):
    col4, col5, col6 = st.columns(3)
    agg_df = df.groupby(["Age", "Alzheimer’s Diagnosis"])["Cognitive Test Score"].mean().reset_index()
    with col4:
        fig3 = px.line(agg_df, x="Age", y="Cognitive Test Score", color="Alzheimer’s Diagnosis",
                      markers=True, color_discrete_map={"Yes": "#2ca02c", "No": "#7f7f7f"})
        fig3.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig3, use_container_width=True)
    with col5:
        fig4 = px.scatter(agg_df, x="Age", y="Cognitive Test Score", color="Alzheimer’s Diagnosis",
                         trendline="ols", symbol="Alzheimer’s Diagnosis",
                         color_discrete_map={"Yes": "#2ca02c", "No": "#7f7f7f"})
        fig4.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig4, use_container_width=True)
    with col6:
        fig5 = px.histogram(df, x="Gender", color="Alzheimer’s Diagnosis", barmode="group",
                           color_discrete_map={"Yes": "#2ca02c", "No": "#7f7f7f"})
        fig5.update_layout(height=400, margin=dict(t=30, b=20))
        st.plotly_chart(fig5, use_container_width=True)

# --- Section: Lifestyle Risk Profile & Country Analysis --- #
with st.expander("🌍 Lifestyle Risk Profile & Country Analysis", expanded=False):
    col7, col8, col9 = st.columns(3)
    with col7:
        lifestyle_cols = ["Smoking Status", "Alcohol Consumption", "Physical Activity Level", "Sleep Quality", "Dietary Habits"]
        radar_data = {}
        for group in ["Yes", "No"]:
            subset = df[df["Alzheimer’s Diagnosis"] == group]
            counts = [subset[col].value_counts(normalize=True).max() for col in lifestyle_cols]
            radar_data[group] = counts
        fig6 = go.Figure()
        for group, values in radar_data.items():
            fig6.add_trace(go.Scatterpolar(
                r=values,
                theta=lifestyle_cols,
                fill='toself',
                name=f"Diagnosis: {group}"
            ))
        fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          title="Lifestyle Risk Profile by Diagnosis", height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig6, use_container_width=True)
    with col8:
        diagnosed_df = df[df["Alzheimer’s Diagnosis"] == "Yes"]
        top_countries = diagnosed_df["Country"].value_counts().head(10).reset_index()
        top_countries.columns = ["Country", "Diagnosed Cases"]
        fig7 = px.pie(top_countries, names="Country", values="Diagnosed Cases",
                     color_discrete_sequence=px.colors.sequential.Greens,
                     title="Top 10 Countries – Alzheimer’s Cases")
        fig7.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig7, use_container_width=True)
    with col9:
        country_counts = diagnosed_df["Country"].value_counts().reset_index()
        country_counts.columns = ["Country", "Diagnosed Cases"]
        fig8 = px.choropleth(
            country_counts,
            locations="Country",
            locationmode="country names",
            color="Diagnosed Cases",
            color_continuous_scale="Greens",
            title="Alzheimer’s Diagnoses by Country"
        )
        fig8.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig8, use_container_width=True)