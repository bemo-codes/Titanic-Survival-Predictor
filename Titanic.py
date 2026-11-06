#############################################################################
#
#   Importing Required Libraries
#
#############################################################################
from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


#############################################################################
#
#   Application Configuration
#
#############################################################################
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "TitanicDataset.csv"

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


#############################################################################
#
#   User Interface Styling
#
#############################################################################
#############################################################################
#
#   Function name:      inject_custom_css
#   Input:              None
#   Description:        Applies the custom dark navy and gold styling to the Streamlit dashboard.
#   Output:             None
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 4%, rgba(18, 111, 140, .18), transparent 28%),
                radial-gradient(circle at 88% 8%, rgba(238, 157, 65, .12), transparent 25%),
                linear-gradient(135deg, #08111f 0%, #101b2d 55%, #111827 100%);
            color: #edf2f7;
        }
        [data-testid="stSidebar"] {
            background: #0b1422;
            border-right: 1px solid rgba(148, 163, 184, .18);
        }
        .hero {
            padding: 30px 32px;
            border: 1px solid rgba(148, 163, 184, .20);
            border-radius: 22px;
            background: linear-gradient(115deg, rgba(14, 42, 61, .92), rgba(28, 38, 57, .88));
            box-shadow: 0 16px 40px rgba(0, 0, 0, .22);
            margin-bottom: 22px;
        }
        .hero-kicker {
            color: #f4b860;
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .hero-title {
            color: #f8fafc;
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.08;
            font-weight: 900;
            letter-spacing: -1.3px;
            margin: 0;
        }
        .hero-copy {
            color: #cbd5e1;
            font-size: 1.02rem;
            max-width: 850px;
            margin-top: 12px;
            margin-bottom: 0;
            line-height: 1.65;
        }
        .section-note {
            color: #9fb0c4;
            font-size: .92rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(17, 31, 49, .78);
            border: 1px solid rgba(148, 163, 184, .17);
            padding: 16px 18px;
            border-radius: 16px;
        }
        div[data-testid="stMetricLabel"] {
            color: #a9bad0;
        }
        .stButton button, .stDownloadButton button {
            border: 0;
            border-radius: 11px;
            font-weight: 750;
            background: linear-gradient(90deg, #e9a23b, #f0c16d);
            color: #172033;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            border: 0;
            color: #101827;
            filter: brightness(1.08);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(26, 41, 61, .65);
            border-radius: 10px 10px 0 0;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(233, 162, 59, .16) !important;
        }
        .result-card {
            border-radius: 18px;
            padding: 22px;
            background: rgba(17, 31, 49, .86);
            border: 1px solid rgba(148, 163, 184, .22);
        }
        .small-muted {
            color: #a8b6c8;
            font-size: .86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


#############################################################################
#
#   Dataset Loading & Data Preprocessing
#
#############################################################################
#############################################################################
#
#   Function name:      load_dataset
#   Input:              path (string)
#   Description:        Loads the Titanic dataset from a CSV file and caches it for efficient Streamlit execution.
#   Output:             DataFrame
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


#############################################################################
#
#   Function name:      prepare_features
#   Input:              data (DataFrame)
#   Description:        Validates the dataset, handles missing values, removes unused columns, and encodes embarkation features.
#   Output:             Feature DataFrame, target Series, and imputation metadata
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Apply the same preprocessing logic as the original training script."""
    required = {
        "Age", "Fare", "Sex", "sibsp", "Parch",
        "Pclass", "Embarked", "Survived"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    working = data.copy()

    # Remove identifiers and the constant column; neither is useful for prediction.
    working = working.drop(
        columns=["Passengerid", "zero", "name"],
        errors="ignore",
    )

    # Ensure numeric columns are represented consistently.
    numeric_columns = ["Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked"]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    # Fill missing values using the same median/mode approach as the source script.
    imputation = {}
    for column in ["Age", "Fare"]:
        median_value = float(working[column].median())
        working[column] = working[column].fillna(median_value)
        imputation[column] = median_value

    embarked_mode = float(working["Embarked"].mode().iloc[0])
    working["Embarked"] = working["Embarked"].fillna(embarked_mode)
    imputation["Embarked"] = embarked_mode

    for column in ["Sex", "sibsp", "Parch", "Pclass"]:
        mode_value = float(working[column].mode().iloc[0])
        working[column] = working[column].fillna(mode_value)
        imputation[column] = mode_value

    y = working["Survived"].astype(int)
    X = working.drop(columns=["Survived"])

    # One-hot encode the already-numeric embarkation codes.
    X = pd.get_dummies(
        X,
        columns=["Embarked"],
        drop_first=True,
        dtype=int,
    )

    return X, y, imputation


#############################################################################
#
#   Function name:      train_models
#   Input:              path (string), tree_max_depth (integer)
#   Description:        Preprocesses the Titanic data, creates one shared train-test split,
#                       trains Logistic Regression and Decision Tree classifiers, and
#                       calculates evaluation metrics for both models.
#   Output:             Dictionary containing both trained models and evaluation results
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_resource
def train_models(path: str, tree_max_depth: int = 3) -> dict:
    raw_data = pd.read_csv(path)
    X, y, imputation = prepare_features(raw_data)

    # Both algorithms use the exact same split for a fair comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_definitions = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=tree_max_depth,
            random_state=42,
        ),
    }

    model_results = {}
    for model_name, model in model_definitions.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        model_results[model_name] = {
            "model": model,
            "predictions": predictions,
            "probabilities": model.predict_proba(X_test)[:, 1],
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            ),
            "report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Did not survive", "Survived"],
                output_dict=True,
                zero_division=0,
            ),
        }

    return {
        "models": model_results,
        "feature_columns": X.columns.tolist(),
        "imputation": imputation,
        "X_test": X_test,
        "y_test": y_test,
        "class_counts": y.value_counts().sort_index().to_dict(),
        "tree_max_depth": tree_max_depth,
        "training_records": len(X_train),
        "test_records": len(X_test),
    }


#############################################################################
#
#   Function name:      prepare_passenger_input
#   Input:              passenger (dict), feature_columns (list), imputation (dict)
#   Description:        Converts passenger form inputs into the same feature layout used during model training.
#   Output:             Single-row DataFrame aligned with the training features
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def prepare_passenger_input(
    passenger: dict,
    feature_columns: list[str],
    imputation: dict,
) -> pd.DataFrame:
    """Convert a form submission into the same feature layout used for training."""
    embarked_value = float(
        passenger.get("Embarked", imputation.get("Embarked", 0))
    )

    # The training set creates one-hot columns for the non-baseline
    # embarkation codes (the baseline is the lowest code, 0).
    row_values = {
        "Age": float(passenger.get("Age", imputation.get("Age", 0))),
        "Fare": float(passenger.get("Fare", imputation.get("Fare", 0))),
        "Sex": float(passenger.get("Sex", imputation.get("Sex", 0))),
        "sibsp": float(passenger.get("sibsp", imputation.get("sibsp", 0))),
        "Parch": float(passenger.get("Parch", imputation.get("Parch", 0))),
        "Pclass": float(passenger.get("Pclass", imputation.get("Pclass", 3))),
    }

    for column in feature_columns:
        if column.startswith("Embarked_"):
            try:
                category_code = float(column.split("_", 1)[1])
                row_values[column] = int(embarked_value == category_code)
            except ValueError:
                row_values[column] = 0

    # Match the exact training columns and their order.
    return pd.DataFrame([row_values]).reindex(
        columns=feature_columns,
        fill_value=0,
    )


#############################################################################
#
#   Function name:      model_download_bytes
#   Input:              bundle (dict)
#   Description:        Serializes both trained classifiers and their shared preprocessing
#                       metadata into a downloadable Joblib model bundle.
#   Output:             bytes
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def model_download_bytes(bundle: dict) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(
        {
            "models": {
                model_name: result["model"]
                for model_name, result in bundle["models"].items()
            },
            "feature_columns": bundle["feature_columns"],
            "imputation": bundle["imputation"],
            "target_mapping": {0: "Did not survive", 1: "Survived"},
            "sex_encoding": {"Male": 0, "Female": 1},
            "embarked_codes": [0, 1, 2],
            "decision_tree_max_depth": bundle["tree_max_depth"],
        },
        buffer,
    )
    return buffer.getvalue()


#############################################################################
#
#   Data Visualization Helper Functions
#
#############################################################################
#############################################################################
#
#   Function name:      survival_count_chart
#   Input:              data (DataFrame)
#   Description:        Creates a chart comparing the number of passengers who survived and did not survive.
#   Output:             Matplotlib Figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def survival_count_chart(data: pd.DataFrame) -> plt.Figure:
    chart_data = data["Survived"].map(
        {0: "Did not survive", 1: "Survived"}
    ).value_counts().reindex(["Did not survive", "Survived"])

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#111b2b")
    ax.set_facecolor("#111b2b")
    bars = ax.bar(
        chart_data.index,
        chart_data.values,
        color=["#64748b", "#e9a23b"],
        width=0.58,
    )
    ax.set_title("Passenger Survival Breakdown", color="#f8fafc", weight="bold", pad=14)
    ax.set_ylabel("Number of passengers", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#475569")
    ax.grid(axis="y", color="#334155", alpha=0.45)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{int(bar.get_height()):,}",
            ha="center",
            color="#f8fafc",
            weight="bold",
        )
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      confusion_matrix_chart
#   Input:              matrix (NumPy array)
#   Description:        Creates a heatmap visualization of the model confusion matrix.
#   Output:             Matplotlib Figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def confusion_matrix_chart(
    matrix: np.ndarray,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor("#111b2b")
    ax.set_facecolor("#111b2b")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        cbar=False,
        linewidths=1,
        linecolor="#233247",
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
        ax=ax,
    )
    ax.set_title(title, color="#f8fafc", weight="bold", pad=14)
    ax.set_xlabel("Predicted label", color="#cbd5e1")
    ax.set_ylabel("Actual label", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1", rotation=0)
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      survival_by_group_chart
#   Input:              data (DataFrame), group_column (string), title (string)
#   Description:        Generates a bar chart showing the survival rate across passenger groups.
#   Output:             Matplotlib Figure
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def survival_by_group_chart(data: pd.DataFrame, group_column: str, title: str) -> plt.Figure:
    grouped = (
        data.groupby(group_column, dropna=False)["Survived"]
        .mean()
        .mul(100)
        .reset_index(name="Survival rate")
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#111b2b")
    ax.set_facecolor("#111b2b")
    sns.barplot(
        data=grouped,
        x=group_column,
        y="Survival rate",
        color="#e9a23b",
        ax=ax,
    )
    ax.set_title(title, color="#f8fafc", weight="bold", pad=14)
    ax.set_ylabel("Survival rate (%)", color="#cbd5e1")
    ax.set_xlabel(group_column, color="#cbd5e1")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#cbd5e1")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#475569")
    ax.grid(axis="y", color="#334155", alpha=0.45)
    fig.tight_layout()
    return fig


#############################################################################
#
#   Main Streamlit Application
#
#############################################################################
#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Runs the Streamlit dashboard, loads passenger data, trains and
#                       compares Logistic Regression and Decision Tree models, and
#                       provides interactive passenger survival predictions.
#   Output:             None
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main() -> None:
    inject_custom_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine Learning • Logistic Regression + Decision Tree</div>
            <h1 class="hero-title">🚢 Titanic Survival Predictor</h1>
            <p class="hero-copy">
                Explore passenger data, compare two classification algorithms,
                inspect model performance, and test how passenger details influence
                estimated survival predictions. Two models, one historic dataset,
                and a dashboard built to make the comparison easy to understand.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Place `TitanicDataset.csv` in the same directory as `app.py`."
        )
        st.stop()

    if not hasattr(st, "session_state"):
        st.error("Streamlit session state is unavailable. Update Streamlit and try again.")
        st.stop()

    # Model configuration is visible in the sidebar.
    with st.sidebar:
        st.markdown("## ⚙️ Model Configuration")
        tree_max_depth = st.slider(
            "Decision Tree maximum depth",
            min_value=1,
            max_value=15,
            value=3,
            step=1,
            help="Controls the maximum depth of the Decision Tree. Logistic Regression is unchanged.",
        )
        st.caption("Both models use the same 80/20 train-test split and random state 42.")

    try:
        data = load_dataset(str(DATA_PATH))
        bundle = train_models(str(DATA_PATH), tree_max_depth)
    except Exception as exc:
        st.error(f"Could not load the dataset or train the models: {exc}")
        st.stop()

    required_display_columns = {"Survived", "Age", "Fare", "Sex", "Pclass"}
    if not required_display_columns.issubset(data.columns):
        st.error("The dataset does not contain all columns required by this dashboard.")
        st.stop()

    models = bundle["models"]
    logistic_results = models["Logistic Regression"]
    tree_results = models["Decision Tree"]

    survival_rate = float(data["Survived"].mean() * 100)
    survived_count = int((data["Survived"] == 1).sum())
    not_survived_count = int((data["Survived"] == 0).sum())

    with st.sidebar:
        st.divider()
        st.markdown("### 📊 Model Snapshot")
        st.metric("Logistic Regression accuracy", f"{logistic_results['accuracy'] * 100:.1f}%")
        st.metric("Decision Tree accuracy", f"{tree_results['accuracy'] * 100:.1f}%")
        st.caption(f"Decision Tree depth: {tree_max_depth}")
        st.divider()
        st.markdown("### 🧾 Dataset notes")
        st.caption("Target: `Survived`")
        st.caption("0 = Did not survive")
        st.caption("1 = Survived")
        st.caption("Sex encoding: 0 = Male, 1 = Female")
        st.caption("Embarked is stored as numeric codes (0, 1, 2).")
        st.divider()
        st.caption("Train-test split: 80% / 20%")
        st.caption("Random state: 42")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Passengers", f"{len(data):,}")
    metric_columns[1].metric("Survived", f"{survived_count:,}")
    metric_columns[2].metric("Did not survive", f"{not_survived_count:,}")
    metric_columns[3].metric("Dataset survival rate", f"{survival_rate:.1f}%")

    tab_overview, tab_explorer, tab_model, tab_predict = st.tabs(
        [
            "🧭 Voyage Overview",
            "🔎 Passenger Explorer",
            "📈 Model Comparison",
            "🔮 Survival Predictor",
        ]
    )

    # -----------------------------------------------------------------
    # Tab 1: Overview
    # -----------------------------------------------------------------
    with tab_overview:
        st.subheader("The passenger manifest")
        st.markdown(
            '<p class="section-note">A quick look at the data behind both models.</p>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Dataset preview")
            st.dataframe(data.head(10), use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Survival breakdown")
            st.pyplot(survival_count_chart(data), use_container_width=True)

        st.markdown("#### Dataset health")
        health_left, health_right = st.columns(2)
        with health_left:
            st.metric("Rows", f"{data.shape[0]:,}")
            st.metric("Columns", f"{data.shape[1]}")
        with health_right:
            total_missing = int(data.isna().sum().sum())
            st.metric("Missing values", f"{total_missing:,}")
            st.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

        with st.expander("View missing values by column"):
            missing_table = (
                data.isna()
                .sum()
                .rename("Missing values")
                .to_frame()
                .sort_values("Missing values", ascending=False)
            )
            st.dataframe(missing_table, use_container_width=True)

        st.info(
            "Both models use the same preprocessed features and the same train-test split. "
            "Passenger identifiers and the constant `zero` column are excluded. "
            "Age and Fare are median-imputed; Embarked is mode-imputed and one-hot encoded."
        )

    # -----------------------------------------------------------------
    # Tab 2: Passenger explorer
    # -----------------------------------------------------------------
    with tab_explorer:
        st.subheader("What patterns show up in the manifest?")
        st.markdown(
            '<p class="section-note">Explore survival rates across passenger groups. '
            'These are descriptive statistics, not proof of cause and effect.</p>',
            unsafe_allow_html=True,
        )

        class_col, sex_col = st.columns(2)
        with class_col:
            st.pyplot(
                survival_by_group_chart(data, "Pclass", "Survival rate by passenger class"),
                use_container_width=True,
            )
        with sex_col:
            sex_data = data.copy()
            sex_data["Sex"] = sex_data["Sex"].map(
                {0: "Male (code 0)", 1: "Female (code 1)"}
            ).fillna(sex_data["Sex"].astype(str))
            st.pyplot(
                survival_by_group_chart(sex_data, "Sex", "Survival rate by sex"),
                use_container_width=True,
            )

        st.markdown("#### Age distribution")
        age_plot_data = data.copy()
        age_plot_data["Outcome"] = age_plot_data["Survived"].map(
            {0: "Did not survive", 1: "Survived"}
        )
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor("#111b2b")
        ax.set_facecolor("#111b2b")
        sns.histplot(
            data=age_plot_data,
            x="Age",
            hue="Outcome",
            bins=24,
            element="step",
            stat="count",
            common_norm=False,
            palette={"Did not survive": "#64748b", "Survived": "#e9a23b"},
            ax=ax,
        )
        ax.set_xlabel("Age", color="#cbd5e1")
        ax.set_ylabel("Passenger count", color="#cbd5e1")
        ax.set_title("Passenger age distribution", color="#f8fafc", weight="bold")
        ax.tick_params(colors="#cbd5e1")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#334155", alpha=0.35)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "The source dataset stores Sex as 0/1 and Embarked as 0/1/2. "
            "The dashboard uses the common Sex encoding shown in the sidebar; "
            "the original embarkation codes are retained without assigning port names."
        )

    # -----------------------------------------------------------------
    # Tab 3: Model comparison
    # -----------------------------------------------------------------
    with tab_model:
        st.subheader("Two algorithms. One dataset. Which performs better?")
        st.markdown(
            '<p class="section-note">Both classifiers are evaluated on exactly the same held-out '
            'test records. Compare more than accuracy, especially because the target classes are imbalanced.</p>',
            unsafe_allow_html=True,
        )

        comparison_rows = []
        for model_name, result in models.items():
            survivor_metrics = result["report"]["Survived"]
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": result["accuracy"],
                    "Balanced accuracy": result["balanced_accuracy"],
                    "Survivor precision": survivor_metrics["precision"],
                    "Survivor recall": survivor_metrics["recall"],
                    "Survivor F1-score": survivor_metrics["f1-score"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows).set_index("Model")

        st.markdown("#### Performance comparison")
        st.dataframe(
            comparison_df.style.format("{:.1%}"),
            use_container_width=True,
        )
        st.caption(
            "Balanced accuracy is the average recall across the two classes. "
            "Survivor precision, recall and F1-score focus specifically on passengers labelled as survived."
        )

        lr_col, dt_col = st.columns(2)
        for column, model_name, result in [
            (lr_col, "Logistic Regression", logistic_results),
            (dt_col, "Decision Tree", tree_results),
        ]:
            with column:
                st.markdown(f"### {model_name}")
                st.metric("Accuracy", f"{result['accuracy'] * 100:.2f}%")
                st.metric("Balanced accuracy", f"{result['balanced_accuracy'] * 100:.2f}%")
                st.markdown("#### Confusion matrix")
                st.pyplot(
                    confusion_matrix_chart(
                        result["confusion_matrix"],
                        title=f"{model_name} Confusion Matrix",
                    ),
                    use_container_width=True,
                )
                st.markdown("#### Classification report")
                report_rows = {
                    key: value
                    for key, value in result["report"].items()
                    if isinstance(value, dict)
                }
                report_df = pd.DataFrame(report_rows).T
                report_df = report_df.rename(
                    index={
                        "0": "Did not survive",
                        "1": "Survived",
                        "macro avg": "Macro average",
                        "weighted avg": "Weighted average",
                    }
                )
                st.dataframe(report_df.round(3), use_container_width=True)

        st.markdown("#### Download both trained models")
        st.caption(
            "The bundle includes both fitted classifiers and the shared feature-column "
            "and imputation metadata needed to prepare prediction inputs."
        )
        st.download_button(
            label="⬇️ Download model bundle (.joblib)",
            data=model_download_bytes(bundle),
            file_name="TitanicSurvivalModels.joblib",
            mime="application/octet-stream",
        )

        st.info(
            f"Decision Tree maximum depth is currently set to {tree_max_depth}. "
            "Changing this setting retrains the tree. It does not change Logistic Regression. "
            "A higher accuracy on one split does not automatically mean a model generalizes better."
        )

    # -----------------------------------------------------------------
    # Tab 4: Prediction
    # -----------------------------------------------------------------
    with tab_predict:
        st.subheader("Put a passenger's details to the test")
        st.markdown(
            '<p class="section-note">Choose a classifier, enter passenger information, '
            "and view the model's estimated survival probability.</p>",
            unsafe_allow_html=True,
        )

        with st.form("passenger_prediction_form"):
            selected_model_name = st.selectbox(
                "Choose prediction model",
                options=["Logistic Regression", "Decision Tree"],
                help="Choose which trained classifier should generate the prediction.",
            )
            col_a, col_b = st.columns(2)

            with col_a:
                passenger_class = st.selectbox(
                    "Passenger class",
                    options=[1, 2, 3],
                    index=2,
                    format_func=lambda value: f"{value}st Class" if value == 1 else (
                        f"{value}nd Class" if value == 2 else "3rd Class"
                    ),
                )
                age = st.number_input(
                    "Age",
                    min_value=0.0,
                    max_value=100.0,
                    value=29.0,
                    step=1.0,
                )
                sex_label = st.selectbox(
                    "Sex",
                    options=["Female", "Male"],
                    index=0,
                )
                siblings_spouses = st.number_input(
                    "Siblings / spouses aboard (SibSp)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )

            with col_b:
                fare = st.number_input(
                    "Fare",
                    min_value=0.0,
                    max_value=600.0,
                    value=32.0,
                    step=1.0,
                )
                parents_children = st.number_input(
                    "Parents / children aboard (Parch)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )
                embarked_code = st.selectbox(
                    "Embarkation code",
                    options=[0, 1, 2],
                    index=2,
                    help="The supplied CSV stores Embarked as numeric codes. Choose the code used in that dataset.",
                )

            submitted = st.form_submit_button(
                "🔮 Predict survival",
                use_container_width=True,
            )

        if submitted:
            sex_code = 1 if sex_label == "Female" else 0
            passenger = {
                "Age": age,
                "Fare": fare,
                "Sex": sex_code,
                "sibsp": siblings_spouses,
                "Parch": parents_children,
                "Pclass": passenger_class,
                "Embarked": float(embarked_code),
            }

            try:
                prediction_frame = prepare_passenger_input(
                    passenger,
                    bundle["feature_columns"],
                    bundle["imputation"],
                )
                selected_model = models[selected_model_name]["model"]
                prediction = int(selected_model.predict(prediction_frame)[0])
                survival_probability = float(
                    selected_model.predict_proba(prediction_frame)[0][1]
                )

                st.divider()
                result_left, result_right = st.columns([1, 1])

                with result_left:
                    st.markdown(f"#### {selected_model_name} prediction")
                    if prediction == 1:
                        st.success("The model predicts: **Survived**")
                    else:
                        st.warning("The model predicts: **Did not survive**")

                    st.metric(
                        "Estimated survival probability",
                        f"{survival_probability * 100:.1f}%",
                    )
                    st.progress(
                        min(max(survival_probability, 0.0), 1.0),
                        text="Model-estimated probability",
                    )

                with result_right:
                    st.markdown("#### Passenger details")
                    details = pd.DataFrame(
                        {
                            "Input": [
                                "Prediction model",
                                "Passenger class",
                                "Age",
                                "Sex",
                                "Fare",
                                "SibSp",
                                "Parch",
                                "Embarked code",
                            ],
                            "Value": [
                                selected_model_name,
                                passenger_class,
                                age,
                                sex_label,
                                f"{fare:.2f}",
                                siblings_spouses,
                                parents_children,
                                embarked_code,
                            ],
                        }
                    )
                    st.dataframe(details, use_container_width=True, hide_index=True)

                st.caption(
                    f"This is a prediction from the {selected_model_name} classifier trained "
                    "on the supplied historical dataset. It is an educational demonstration, "
                    "not a statement about an individual's actual fate."
                )
            except Exception as exc:
                st.error(f"Prediction could not be generated: {exc}")

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#9fb0c4; padding:8px 0 18px 0;">
            Built with Python 🐍 · Streamlit ⚡ · Scikit-learn 🤖
            <br><span class="small-muted">An educational machine learning project by Shivam Ramesh Kurlekar</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


#############################################################################
#
#   Application Entry Point
#
#############################################################################
if __name__ == "__main__":
    main()
