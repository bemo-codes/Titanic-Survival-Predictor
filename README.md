# 🚢 Titanic Survival Predictor — Interactive Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-6C63FF)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic_Regression-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployable-2EA44F)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

**Could a machine learning model estimate whether a Titanic passenger survived, using only a handful of passenger details?**

This project turns a classic classification problem into an interactive web application. The **Titanic Survival Predictor** uses **Logistic Regression** to estimate a passenger's survival outcome from details such as age, fare, sex, passenger class, family members aboard, and embarkation code.

The dashboard brings the full workflow into one place: explore the dataset, inspect model performance, compare passenger survival patterns, and submit custom passenger details for a prediction.

> This is an educational machine learning demonstration based on historical data. Its predictions are statistical estimates, not certainty about any individual.

---

## ✨ What can you do with it?

### 🧭 Voyage Overview
- Explore the passenger dataset.
- View the number of passengers, survival counts, and overall survival rate.
- Preview passenger records.
- Inspect missing values and duplicate rows.
- Understand the preprocessing applied before model training.

### 🔎 Passenger Explorer
- Compare survival rates across passenger classes.
- Explore survival rates by the encoded sex category.
- Visualize passenger age distribution.
- Review patterns in the historical dataset.

### 📈 Model Performance
- Train a Logistic Regression classifier.
- Evaluate the model on a held-out test set.
- View accuracy and a confusion matrix.
- Inspect precision, recall, and F1-score.
- Download the trained model bundle with its feature and imputation metadata.

### 🔮 Survival Predictor
- Enter custom passenger information.
- Select passenger class and sex.
- Adjust age, fare, family counts, and embarkation code.
- Get a predicted class and estimated survival probability.

---

## 🧠 Machine Learning Workflow

The application follows the same core Logistic Regression workflow as the original training script, presented through a Streamlit interface.

### 1. Dataset

The project uses `TitanicDataset.csv`, containing 1,309 passenger records and 10 columns.

| Column | Role |
|---|---|
| `Passengerid` | Passenger identifier; excluded from training |
| `Age` | Passenger age |
| `Fare` | Ticket fare |
| `Sex` | Encoded sex feature |
| `sibsp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `zero` | Constant column; excluded from training |
| `Pclass` | Passenger class |
| `Embarked` | Encoded embarkation category |
| `Survived` | Target: 0 = did not survive, 1 = survived |

### 2. Data Preprocessing

The application:
- Drops `Passengerid`, `zero`, and `name` when present.
- Fills missing `Age` and `Fare` values using their respective medians.
- Fills missing `Embarked` values using the mode.
- Fills missing values in the remaining model input columns using their mode.
- Converts `Embarked` into dummy variables using one-hot encoding with `drop_first=True`.

The source dataset already stores `Sex` and `Embarked` as numeric codes. The prediction form uses the common `Sex` mapping of 0 = Male and 1 = Female. Embarkation is kept as codes 0, 1, and 2 rather than assigning port names that are not included in the CSV.

### 3. Train-Test Split

The processed dataset is divided into:
- 80% training data
- 20% testing data

The split uses `random_state=42` so that the same split can be reproduced.

### 4. Model Training

The application uses Scikit-learn's `LogisticRegression` with `max_iter=1000`.

The model learns the relationship between the passenger features and the target variable `Survived`.

### 5. Evaluation

Model performance is measured using:
- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

The app calculates these metrics when the model is trained and displays them in the Model Performance tab.

**Baseline from the supplied dataset:** with the original 80/20 split and `random_state=42`, the current Logistic Regression workflow obtains approximately **76.72% accuracy**. In that run, recall for the `Survived` class is approximately **37.0%**. This matters: accuracy alone hides that the model misses a substantial number of actual survivors. Use the complete classification report, not just the headline accuracy, when judging the model.

### 6. Prediction

For a new passenger, the application:
1. Collects the passenger details from the form.
2. Applies the same missing-value and encoding conventions used for training.
3. Aligns the input with the model's training feature columns.
4. Predicts the class.
5. Displays the model-estimated probability of survival.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and preprocessing |
| NumPy | Numerical data handling |
| Scikit-learn | Machine learning and evaluation |
| Logistic Regression | Binary classification algorithm |
| Matplotlib | Charts and visualizations |
| Seaborn | Statistical visualizations |
| Joblib | Model bundle serialization |

---

## 📁 Project Structure

```text
Titanic-Survival-Predictor/
│
├── app.py                 # Streamlit application
├── TitanicDataset.csv     # Cleanly renamed dataset
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git
├── README.md              # Project documentation
├── LICENSE                # MIT License
│
└── .streamlit/
    └── config.toml        # Streamlit theme configuration
```

---

## ⚙️ Run Locally

### Prerequisites
- Python 3.10 or newer
- Git
- pip

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Titanic-Survival-Predictor
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Streamlit normally uses:

```text
http://localhost:8501
```

---

## 🧪 Example Prediction Inputs

Try a few different combinations to see how the model responds:

| Input | Example A | Example B |
|---|---:|---:|
| Passenger class | 1 | 3 |
| Age | 28 | 35 |
| Sex | Female | Male |
| Fare | 75.00 | 8.05 |
| SibSp | 0 | 1 |
| Parch | 0 | 0 |
| Embarked code | 0 | 2 |

These are example inputs for experimenting with the model, not guaranteed outcomes.

---

## ⚠️ Important Notes & Limitations

- The model learns patterns from the supplied dataset and may not generalize to every passenger or scenario.
- The source dataset contains encoded categorical values; the original labels for embarkation codes are not included in the supplied CSV.
- The user-facing sex mapping is 0 = Male and 1 = Female.
- The current Logistic Regression model uses the source workflow without feature scaling.
- In the supplied baseline split, recall for the `Survived` class is relatively low (about 37%), so the model misses many actual survivors despite its approximately 76.7% overall accuracy.
- A single 80/20 train-test split is used; cross-validation is not currently implemented.
- The dataset is not balanced across the two target classes.
- The estimated probability is a model output, not a guarantee or a causal explanation.
- This is an educational project, not a real-world decision-making tool.

---

## 🔭 Possible Next Improvements

- Add cross-validation and compare multiple train-test splits.
- Compare Logistic Regression with Random Forest, Decision Tree, and Gradient Boosting.
- Add feature scaling and evaluate whether it improves performance.
- Add ROC curves, AUC, and precision-recall visualizations.
- Add feature importance or model explainability using coefficients and SHAP.
- Add a more detailed passenger data explorer.
- Add automated model comparison and hyperparameter tuning.
- Add dataset validation and support for alternate Titanic CSV files.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📜 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  Built with Python 🐍 · Streamlit ⚡ · Scikit-learn 🤖
</p>
<p align="center">
  <i>Turning passenger data into a machine learning story.</i>
</p>
