# 🚢 Titanic Survival Predictor — Interactive Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-6C63FF)
![Logistic Regression](https://img.shields.io/badge/Model-Logistic_Regression-2E86AB)
![Decision Tree](https://img.shields.io/badge/Model-Decision_Tree-2E8B57)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

**Could a machine learning model estimate whether a Titanic passenger survived, using only a handful of passenger details?**

This project turns the classic Titanic classification problem into an interactive Streamlit application. It trains and compares two supervised machine learning algorithms: **Logistic Regression** and **Decision Tree**. Both models use passenger details such as age, fare, sex, passenger class, family members aboard, and embarkation code to predict the target variable `Survived`.

The dashboard brings the workflow into one place: explore the dataset, inspect passenger survival patterns, compare model performance, adjust the Decision Tree depth, and submit custom passenger details for a prediction from either model.

> This is an educational machine learning demonstration based on historical data. Its predictions are statistical estimates, not certainty about any individual.

---

## ✨ What can you do with it?

### 🧭 Voyage Overview
- Explore the passenger dataset.
- View the passenger count, survival counts, and overall survival rate.
- Preview passenger records.
- Inspect missing values and duplicate rows.
- Review the preprocessing applied before model training.

### 🔎 Passenger Explorer
- Compare survival rates across passenger classes.
- Explore survival rates by the encoded sex category.
- Visualize passenger age distribution.
- Review patterns in the historical dataset.

### 📈 Model Comparison
- Train Logistic Regression and Decision Tree classifiers on the same train-test split.
- Compare accuracy and balanced accuracy.
- Compare survivor precision, recall, and F1-score.
- Inspect a separate confusion matrix and classification report for each model.
- Adjust the Decision Tree maximum depth from the sidebar.
- Download both trained models and their preprocessing metadata as a Joblib bundle.

### 🔮 Survival Predictor
- Choose Logistic Regression or Decision Tree for prediction.
- Enter custom passenger information.
- Select passenger class and sex.
- Adjust age, fare, family counts, and embarkation code.
- View the predicted class and estimated survival probability from the selected classifier.

---

## 🧠 Machine Learning Workflow

The application uses the same preprocessing pipeline and train-test split for both classifiers so their evaluation results can be compared on the same held-out records.

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
- Fills missing values in the other model input columns using their mode.
- Converts `Embarked` into dummy variables using one-hot encoding with `drop_first=True`.

The source dataset stores `Sex` and `Embarked` as numeric codes. The prediction form uses the common `Sex` mapping of 0 = Male and 1 = Female. Embarkation is kept as codes 0, 1, and 2 rather than assigning port names that are not included in the supplied CSV.

### 3. Train-Test Split

The processed dataset is divided into:
- 80% training data
- 20% testing data

The split uses `random_state=42`, giving both models the same training and test records for a direct comparison.

### 4. Logistic Regression

Logistic Regression is trained using Scikit-learn's `LogisticRegression` with `max_iter=1000`.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

This model estimates the probability of a passenger belonging to the survived class based on a learned linear decision boundary.

### 5. Decision Tree

The second classifier is Scikit-learn's `DecisionTreeClassifier`.

```python
model = DecisionTreeClassifier(
    max_depth=tree_max_depth,
    random_state=42,
)
model.fit(X_train, y_train)
```

The maximum depth is adjustable from the Streamlit sidebar. The depth setting controls how many decision levels the tree can build. A shallow tree is easier to interpret, while a deeper tree can learn more detailed patterns and may overfit.

### 6. Model Evaluation

Both models are evaluated on the same test set using:
- **Accuracy:** Overall proportion of correct predictions.
- **Balanced accuracy:** Average recall across the two classes; useful when class counts differ.
- **Precision:** Of the passengers predicted as survivors, how many actually survived.
- **Recall:** Of the actual survivors, how many the model identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts correct and incorrect predictions for both classes.

### 7. Prediction

For a new passenger, the application:
1. Collects the passenger details from the form.
2. Applies the same feature layout used during training.
3. Aligns the input with the model's training columns.
4. Uses the selected classifier to predict the class.
5. Displays the model-estimated probability of survival.

---

## 📊 Baseline Model Results

With the supplied CSV, an 80/20 split, `random_state=42`, and Decision Tree `max_depth=3`, the baseline results are:

| Metric | Logistic Regression | Decision Tree (depth 3) |
|---|---:|---:|
| Accuracy | 76.72% | 78.63% |
| Balanced accuracy | 64.52% | 67.95% |
| Survivor precision | 64.29% | 68.09% |
| Survivor recall | 36.99% | 43.84% |
| Survivor F1-score | 46.96% | 53.33% |

These are results from one fixed train-test split, not a guarantee of future performance. The dataset is imbalanced, so accuracy should not be considered alone. The survivor recall values show that both baseline models miss a substantial proportion of actual survivors. The dashboard reports the full classification metrics to make this limitation visible.

Changing the Decision Tree depth in the sidebar changes its results. Logistic Regression remains unchanged, and both models continue to use the same split.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and preprocessing |
| NumPy | Numerical data handling |
| Scikit-learn | Model training and evaluation |
| Logistic Regression | Linear classification algorithm |
| Decision Tree | Tree-based classification algorithm |
| Matplotlib | Charts and visualizations |
| Seaborn | Statistical visualizations |
| Joblib | Model bundle serialization |

---

## 📁 Project Structure

```text
Titanic-Survival-Predictor/
│
├── app.py                 # Streamlit application
├── TitanicDataset.csv     # Titanic passenger dataset
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
git clone https://github.com/bemo-codes/Titanic-Survival-Predictor.git
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

Try different combinations to see how the two classifiers respond:

| Input | Example A | Example B |
|---|---:|---:|
| Passenger class | 1 | 3 |
| Age | 28 | 35 |
| Sex | Female | Male |
| Fare | 75.00 | 8.05 |
| SibSp | 0 | 1 |
| Parch | 0 | 0 |
| Embarked code | 0 | 2 |

These are example inputs for experimenting with the models, not guaranteed outcomes.

---

## ⚠️ Important Notes & Limitations

- The models learn patterns from the supplied dataset and may not generalize to every passenger or scenario.
- The source dataset contains encoded categorical values; the original labels for embarkation codes are not included in the supplied CSV.
- The prediction form uses the common sex mapping of 0 = Male and 1 = Female.
- Both classifiers use the current source workflow without feature scaling.
- The baseline dataset is imbalanced across the two target classes.
- A single 80/20 train-test split is used; cross-validation is not currently implemented.
- Decision Tree performance depends on the selected maximum depth and may overfit at greater depths.
- The estimated probability is a model output, not a guarantee or a causal explanation.
- This is an educational project, not a real-world decision-making tool.

---

## 🔭 Possible Next Improvements

- Add stratified cross-validation and compare results across multiple splits.
- Add class-weight tuning and threshold optimization.
- Add feature engineering such as family size and whether a passenger travelled alone.
- Add ROC curves, AUC, and precision-recall visualizations.
- Add feature importance and model explainability.
- Add automated hyperparameter tuning for both algorithms.
- Add further classifiers such as Random Forest and Gradient Boosting.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

<p align="center">
  Built with Python 🐍 · Streamlit ⚡ · Logistic Regression 📈 · Decision Trees 🌳
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
