# Decoding the Psychology of Online vs. In-Store Retail


## 📌 Project Overview
This project explores a dataset of **11,000+ consumers** to identify the psychological and demographic drivers behind shopping preferences (Online vs. In-Store). Unlike basic models that rely on past spending, this project focuses on **psychographic profiling** to understand the *why* behind consumer choices.

The final model successfully navigates **data leakage** and **extreme class imbalance** to provide actionable business insights.

## 🚀 Key Features & Workflow
1. **Exploratory Data Analysis (EDA):** - Invalidated common myths regarding age and income being primary drivers for channel selection.
   - Identified the "Touch & Feel" requirement as the most significant psychological barrier to online migration.
2. **Data Preprocessing:**
   - Handled categorical variables using One-Hot and Label Encoding.
   - Performed stratified splitting to preserve minority class proportions.
3. **Advanced Machine Learning:**
   - Developed a **Random Forest** pipeline.
   - **Feature Selection:** Removed transactional "leakage" variables (like `avg_store_spend`) to build a true predictive profiling tool.
   - **Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to rescue the model from "majority class bias."

## 📊 Business Insights
* **The Sensory Gap:** The `need_touch_feel_score` is the #1 predictor. Businesses should invest in AR/VR or flexible return policies to simulate physical interaction.
* **Logistical Myths:** Delivery speed was found to be less influential than **Delivery Fee Sensitivity**, suggesting that cost-effectiveness outweighs speed for this demographic.
* **Digital Lifestyle:** "Daily Internet Hours" is a more reliable predictor than "Age," proving that digital habits transcend generational boundaries.

## 📈 Model Performance
| Model Version | Accuracy | Online F1-Score | Note |
| :--- | :--- | :--- | :--- |
| **Baseline (With Leakage)** | 95% | 0.87 | Biased by past spending data. |
| **Psychographic (No SMOTE)** | 87% | 0.00 | Failed to detect minority classes. |
| **Balanced (SMOTE)** | **74%** | **0.14** | **Best for Business:** Real predictive power for Online/Hybrid segments. |

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), Seaborn, Matplotlib.
* **Environment:** Jupyter Notebook / Anaconda.
