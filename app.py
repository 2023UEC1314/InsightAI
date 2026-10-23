
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Page configuration
st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)

# 2. Application heading
st.title("📊 InsightAI")
st.subheader("AI-Powered Business Intelligence Assistant")
st.write("Upload a CSV to inspect your data and train a machine learning model.")

# 3. Upload CSV file
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty or len(df.columns) == 0:
            st.warning("The uploaded dataset is empty.")

        else:
            # 4. Dataset overview
            st.header("📋 Dataset Overview")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isna().sum().sum()))

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # 5. Machine Learning
            st.header("🤖 Train a Machine Learning Model")

            target = st.selectbox(
                "Choose the column you want to predict",
                df.columns.tolist(),
                key="ml_target"
            )

            task = st.selectbox(
                "Choose the prediction task",
                ["Classification", "Regression"],
                key="ml_task"
            )

            if st.button("Train Model"):
                try:
                    # Separate target and features
                    data = df.dropna(subset=[target]).copy()
                    X = data.drop(columns=[target])
                    y = data[target]

                    # Remove constant or entirely missing features
                    X = X.drop(
                        columns=[
                            col for col in X.columns
                            if X[col].nunique() <= 1
                        ]
                    )

                    if len(data) < 10:
                        st.error("At least 10 usable rows are required.")

                    elif y.nunique() < 2:
                        st.error("The target must contain at least two distinct values.")

                    elif X.shape[1] == 0:
                        st.error("No useful feature columns remain for training.")

                    elif task == "Regression" and not pd.api.types.is_numeric_dtype(y):
                        st.error(
                            "Regression requires a numerical target. "
                            "Choose a numeric column such as Price."
                        )

                    else:
                        # Identify feature types
                        numeric_cols = X.select_dtypes(
                            include="number"
                        ).columns.tolist()

                        categorical_cols = X.select_dtypes(
                            exclude="number"
                        ).columns.tolist()

                        # Preprocessing pipelines
                        transformers = []

                        if numeric_cols:
                            numeric_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="median"))
                            ])

                            transformers.append(
                                ("num", numeric_pipeline, numeric_cols)
                            )

                        if categorical_cols:
                            categorical_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                            ])

                            transformers.append(
                                ("cat", categorical_pipeline, categorical_cols)
                            )

                        preprocessor = ColumnTransformer(transformers)

                        # Select model
                        if task == "Classification":
                            model = RandomForestClassifier(
                                n_estimators=100,
                                random_state=42
                            )
                        else:
                            model = RandomForestRegressor(
                                n_estimators=100,
                                random_state=42
                            )

                        pipeline = Pipeline([
                            ("preprocessing", preprocessor),
                            ("model", model)
                        ])

                        # Split data into training and testing sets
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )

                        # Train and predict
                        pipeline.fit(X_train, y_train)
                        predictions = pipeline.predict(X_test)

                        st.success("Model training completed!")

                        # Display sample predictions
                        st.subheader("Sample Predictions")

                        results = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": predictions
                        })

                        st.dataframe(results, use_container_width=True)

                        # Classification evaluation
                        if task == "Classification":
                            accuracy = accuracy_score(y_test, predictions)

                            st.metric(
                                "Test Accuracy",
                                f"{accuracy:.1%}"
                            )

                            st.subheader("Classification Report")

                            report = classification_report(
                                y_test,
                                predictions,
                                output_dict=True,
                                zero_division=0
                            )

                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(3))

                            st.subheader("Confusion Matrix")

                            fig, ax = plt.subplots(figsize=(6, 4))

                            ConfusionMatrixDisplay.from_predictions(
                                y_test,
                                predictions,
                                ax=ax,
                                cmap="Blues"
                            )

                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        # Regression evaluation
                        else:
                            mae = mean_absolute_error(
                                y_test,
                                predictions
                            )

                            st.metric(
                                "Test Mean Absolute Error (MAE)",
                                f"{mae:,.2f}"
                            )

                except Exception as e:
                    st.error(f"Could not train the model: {e}")

    except (pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError) as e:
        st.error(f"Could not read this CSV file: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("Upload a CSV file to get started.")
