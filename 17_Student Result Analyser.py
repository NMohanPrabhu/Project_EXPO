import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Set page configuration
st.set_page_config(page_title="Student Results Analysis", layout="wide", page_icon="📊")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title(" Student Results Analysis Dashboard")
st.markdown("Upload a CSV file containing student results to view detailed insights, predictions, and generate reports.")

# File uploader
uploaded_file = st.file_uploader("Drag and drop your CSV file here", type=['csv'])

def create_pdf_report(df, top_10_ranked, one_arrear, two_arrears, more_arrears):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Student Results Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Top 10 Ranked Students
    elements.append(Paragraph("Top 10 Ranked Students (No Arrears)", styles['Heading2']))
    data = [df.columns.tolist()] + top_10_ranked.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Arrears Summary
    elements.append(Paragraph("Arrears Summary", styles['Heading2']))
    arrears_data = [
        ["Arrear Count", "Number of Students"],
        ["1 Arrear", len(one_arrear)],
        ["2 Arrears", len(two_arrears)],
        ["More than 2 Arrears", len(more_arrears)]
    ]
    arrears_table = Table(arrears_data)
    arrears_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(arrears_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        
        # Verify required columns
        required_columns = ['Student Name', 'Total Marks', 'Average', 'Arrears']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain required columns: " + ", ".join(required_columns))
            st.stop()

        # Dynamically identify subject columns (numeric columns excluding required ones)
        subject_columns = [col for col in df.columns if col not in required_columns and pd.api.types.is_numeric_dtype(df[col])]
        if not subject_columns:
            st.error("No valid subject columns found with numeric data.")
            st.stop()

        st.markdown(f"### Detected Subjects: {', '.join(subject_columns)}")
        st.markdown(f"Total number of subjects: {len(subject_columns)}")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs([" Insights", " Visualizations", " Predictions"])

        with tab1:
            # Insights button
            if st.button("Generate Insights", key="insights_button"):
                st.subheader("Detailed Insights")

                # Filter students with no arrears for overall ranking
                no_arrears_df = df[df['Arrears'] == 0].copy()

                # Top 10 Ranked Students (No Arrears)
                st.markdown("###  Top 10 Ranked Students (No Arrears)")
                top_10_ranked = no_arrears_df.sort_values(by='Total Marks', ascending=False).head(10)
                top_10_ranked = top_10_ranked.reset_index(drop=True)
                st.dataframe(
                    top_10_ranked,
                    use_container_width=True,
                    column_config={
                        "Total Marks": st.column_config.NumberColumn(format="%.0f"),
                        "Average": st.column_config.NumberColumn(format="%.2f"),
                        "Arrears": st.column_config.NumberColumn(format="%.0f"),
                        **{subject: st.column_config.NumberColumn(format="%.0f") for subject in subject_columns}
                    }
                )

                # Top 10 Students for Each Subject (Including students with arrears)
                st.markdown("###  Top 10 Students by Subject")
                
                for subject in subject_columns:
                    with st.expander(f"Top 10 in {subject}", expanded=False):
                        top_10_subject = df[['Student Name', subject, 'Total Marks', 'Average', 'Arrears']].sort_values(by=subject, ascending=False).head(10)
                        top_10_subject = top_10_subject.reset_index(drop=True)
                        st.dataframe(
                            top_10_subject,
                            use_container_width=True,
                            column_config={
                                subject: st.column_config.NumberColumn(format="%.0f"),
                                "Total Marks": st.column_config.NumberColumn(format="%.0f"),
                                "Average": st.column_config.NumberColumn(format="%.2f"),
                                "Arrears": st.column_config.NumberColumn(format="%.0f")
                            }
                        )

                # Arrears Analysis
                st.markdown("###  Students with Arrears")
                
                # 1 Arrear
                one_arrear = df[df['Arrears'] == 1][['Student Name'] + subject_columns + ['Total Marks', 'Average']]
                if not one_arrear.empty:
                    st.markdown("#### Students with 1 Arrear")
                    st.dataframe(
                        one_arrear.reset_index(drop=True),
                        use_container_width=True,
                        column_config={
                            "Total Marks": st.column_config.NumberColumn(format="%.0f"),
                            "Average": st.column_config.NumberColumn(format="%.2f"),
                            **{subject: st.column_config.NumberColumn(format="%.0f") for subject in subject_columns}
                        }
                    )
                else:
                    st.info("No students with exactly 1 arrear.")

                # 2 Arrears
                two_arrears = df[df['Arrears'] == 2][['Student Name'] + subject_columns + ['Total Marks', 'Average']]
                if not two_arrears.empty:
                    st.markdown("#### Students with 2 Arrears")
                    st.dataframe(
                        two_arrears.reset_index(drop=True),
                        use_container_width=True,
                        column_config={
                            "Total Marks": st.column_config.NumberColumn(format="%.0f"),
                            "Average": st.column_config.NumberColumn(format="%.2f"),
                            **{subject: st.column_config.NumberColumn(format="%.0f") for subject in subject_columns}
                        }
                    )
                else:
                    st.info("No students with exactly 2 arrears.")

                # More than 2 Arrears
                more_arrears = df[df['Arrears'] > 2][['Student Name'] + subject_columns + ['Total Marks', 'Average', 'Arrears']]
                if not more_arrears.empty:
                    st.markdown("#### Students with More than 2 Arrears")
                    st.dataframe(
                        more_arrears.reset_index(drop=True),
                        use_container_width=True,
                        column_config={
                            "Total Marks": st.column_config.NumberColumn(format="%.0f"),
                            "Average": st.column_config.NumberColumn(format="%.2f"),
                            "Arrears": st.column_config.NumberColumn(format="%.0f"),
                            **{subject: st.column_config.NumberColumn(format="%.0f") for subject in subject_columns}
                        }
                    )
                else:
                    st.info("No students with more than 2 arrears.")

                # PDF Report Generation
                st.markdown("###  Download Report")
                pdf_buffer = create_pdf_report(df, top_10_ranked, one_arrear, two_arrears, more_arrears)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="student_results_report.pdf",
                    mime="application/pdf"
                )

        with tab2:
            st.subheader("Visual Analytics")

            # Average Marks Distribution
            st.markdown("####  Average Marks Distribution")
            fig_avg = px.histogram(df, x='Average', nbins=20, title="Distribution of Average Marks",
                                 color_discrete_sequence=['#4CAF50'])
            fig_avg.update_layout(bargap=0.1, xaxis_title="Average Marks", yaxis_title="Count")
            st.plotly_chart(fig_avg, use_container_width=True)

            # Arrears Distribution
            st.markdown("####  Arrears Distribution")
            arrears_counts = df['Arrears'].value_counts().reset_index()
            arrears_counts.columns = ['Arrears', 'Count']
            fig_arrears = px.bar(arrears_counts, x='Arrears', y='Count', 
                               title="Distribution of Arrears",
                               color_discrete_sequence=['#FF5733'])
            fig_arrears.update_layout(xaxis_title="Number of Arrears", yaxis_title="Number of Students")
            st.plotly_chart(fig_arrears, use_container_width=True)

            # Subject-wise Average Marks
            st.markdown("####  Subject-wise Average Marks")
            subject_means = df[subject_columns].mean().reset_index()
            subject_means.columns = ['Subject', 'Average Mark']
            fig_subjects = px.bar(subject_means, x='Subject', y='Average Mark',
                                title="Average Marks per Subject",
                                color_discrete_sequence=['#2196F3'])
            fig_subjects.update_layout(yaxis_title="Average Mark")
            st.plotly_chart(fig_subjects, use_container_width=True)

        with tab3:
            st.subheader("Predictive Analytics")
            st.markdown("Predict the likelihood of students having arrears based on their marks.")

            # Prepare data for prediction
            X = df[subject_columns]
            y = (df['Arrears'] > 0).astype(int)  # 1 if arrears, 0 if no arrears

            # Split data and train the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train logistic regression model
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)

            # Predict probabilities
            X_scaled = scaler.transform(X)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Add predictions to dataframe
            df['Arrear Probability'] = probabilities
            df['Arrear Risk'] = df['Arrear Probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
            )

            # Display predictions
            st.markdown("###  Arrear Risk Predictions")
            st.dataframe(
                df[['Student Name', 'Arrear Probability', 'Arrear Risk'] + subject_columns].sort_values(by='Arrear Probability', ascending=False),
                use_container_width=True,
                column_config={
                    "Arrear Probability": st.column_config.NumberColumn(format="%.2f"),
                    **{subject: st.column_config.NumberColumn(format="%.0f") for subject in subject_columns}
                }
            )

            # Prediction visualization
            st.markdown("####  Arrear Risk Distribution")
            risk_counts = df['Arrear Risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig_risk = px.bar(risk_counts, x='Risk Level', y='Count',
                            title="Distribution of Arrear Risk Levels",
                            color_discrete_sequence=['#FF5733'])
            fig_risk.update_layout(xaxis_title="Risk Level", yaxis_title="Number of Students")
            st.plotly_chart(fig_risk, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing the CSV file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")