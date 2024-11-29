import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def page1():
    def local_css():
        st.markdown("""
        <style>
        .theme-toggle {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab-list"] button {
            padding: 10px 15px;
            border-radius: 8px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--text-color);
        }
        </style>
        """, unsafe_allow_html=True)

    def load_data():
        try:
            data = pd.read_csv("C:/VS code/IDP/Data/heart.csv")
            return data
        except FileNotFoundError:
            st.error("❌ Dataset not found. Please check the file path.")
            return None

    def create_correlation_matrix(data, selected_features):
        """Create a correlation matrix with detailed annotations"""
        corr = data[selected_features].corr()

        # Create annotated correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',  # Diverging color scale from red to blue
            zmin=-1, 
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            width=800,
            height=800,
            xaxis_title="Features",
            yaxis_title="Features"
        )

        return fig

    def eda_page():
        local_css()
        # Theme toggle (simulate with Streamlit's built-in theming)
        st.markdown("""
        <div class="theme-toggle">
            🌓 Dark/Light Mode
        </div>
        """, unsafe_allow_html=True)

        # Load data
        data = load_data()
        if data is None:
            return

        # Data Overview
        st.markdown("## 📊 Heart Disease Dataset Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(data))

        with col2:
            st.metric("Total Features", len(data.columns) - 1)  # Exclude target

        with col3:
            st.metric("Target Distribution", 
                f"{data['target'].value_counts()[1]} Positive / {data['target'].value_counts()[0]} Negative"
            )

        st.divider()

        # Tabs for Analysis
        tab1, tab2 = st.tabs([
            "📊 Distribution Analysis",
            "🔄 Feature Relationships"
        ])

        # Remove target from feature selection
        all_features = data.columns.tolist()
        all_features.remove('target')

        # Distribution Analysis Tab
        with tab1:
            st.markdown("### 📊 Feature Distributions")

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_features_dist = st.multiselect(
                    "Select features for distribution analysis:",
                    all_features,
                    default=all_features[:3],
                    key="dist_features"
                )

            with col2:
                plot_type = st.selectbox(
                    "Select plot type:",
                    ["Histogram", "Box Plot", "Violin Plot"]
                )

            if st.button("Generate Distribution Plots", key="dist_button"):
                if selected_features_dist:
                    # Create plot based on selected type
                    fig = go.Figure()

                    for feature in selected_features_dist:
                        if plot_type == "Histogram":
                            fig.add_trace(go.Histogram(x=data[feature], name=feature))
                        elif plot_type == "Box Plot":
                            fig.add_trace(go.Box(y=data[feature], name=feature))
                        else:  # Violin Plot
                            fig.add_trace(go.Violin(y=data[feature], name=feature, box_visible=True))

                    fig.update_layout(
                        title=f"{plot_type} of Selected Features",
                        barmode='overlay',
                        xaxis_title="Value",
                        yaxis_title="Frequency",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one feature.")

        # Feature Relationships Tab
        with tab2:
            st.markdown("### 🔄 Feature Relationships")

            # Feature selection for correlation
            selected_features_rel = st.multiselect(
                "Select features for relationship analysis:",
                all_features,
                default=all_features[:5],
                key="rel_features"
            )

            if len(selected_features_rel) > 1:
                # Generate correlation matrix
                fig = create_correlation_matrix(data, selected_features_rel)
                st.plotly_chart(fig, use_container_width=True)

                # Display top correlations
                corr = data[selected_features_rel].corr()
                corr_pairs = []
                for i in range(len(selected_features_rel)):
                    for j in range(i+1, len(selected_features_rel)):
                        corr_pairs.append({
                            'Feature 1': selected_features_rel[i],
                            'Feature 2': selected_features_rel[j],
                            'Correlation': corr.iloc[i, j]
                        })

                # Sort and display correlations
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
                st.markdown("### 🔍 Top Correlations")
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.warning("Please select at least two features for relationship analysis.")

    # Call eda_page function
    eda_page()
