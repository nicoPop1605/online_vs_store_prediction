# Decoding the Psychology of Online vs. In-Store Retail

## 📌 Project Overview
This project explores a dataset of **11,000+ consumers** to identify the psychological and demographic drivers behind shopping preferences. Unlike basic behavioral models that rely on past spending (which often causes data leakage), this project focuses entirely on **psychographic profiling** to understand the *why* behind consumer choices.

The final model successfully navigates **severe class imbalance**, reframes a failing multi-class problem into an actionable binary business segmentation, and utilizes cost-sensitive learning to maximize customer acquisition for digital marketing teams.

## 🚀 Key Features & Workflow
1. **Exploratory Data Analysis (EDA):** - Invalidated common myths regarding age and income being the primary drivers for channel selection.
   - Identified the "Touch & Feel" requirement as the most significant psychological barrier to online migration.
2. **Data Preprocessing & Target Realignment:**
   - Handled categorical variables using Label Encoding.
   - **Business Pivot:** Reframed the original multi-class target (`Store`, `Hybrid`, `Online`) into a Binary classification problem (`Physical` vs. `Digital`) because strictly online and hybrid shoppers exhibited nearly identical psychographic profiles.
3. **Advanced Machine Learning (XGBoost):**
   - **Feature Selection:** Excluded transactional "leakage" variables (like `avg_store_spend`) to build a true predictive profiling tool.
   - **Cost-Sensitive Learning:** Replaced synthetic oversampling (SMOTE) with dynamic class weighting (`scale_pos_weight` in XGBoost) to handle the 6:1 majority class bias without distorting real-world feature distributions.
   - **Probability Threshold Tuning:** Extracted the Precision-Recall curve to dynamically shift the decision boundary from `0.50` to `0.15`. 

## 📊 Business Insights & Strategy

* **The Sensory Gap:** The `need_touch_feel_score` is the #1 predictor. Businesses should invest in AR/VR product visualization or flexible "try-before-you-buy" return policies to simulate physical interaction.
* **Logistical Myths:** Delivery speed was found to be significantly less influential than **Delivery Fee Sensitivity**. To convert physical shoppers, running "Free Shipping" promotions will yield higher ROI than flat-percentage discounts.
* **Digital Lifestyle over Demographics:** "Daily Internet Hours" is a vastly more reliable predictor than "Age," proving that digital shopping habits transcend generational boundaries. Ad spend should be reallocated accordingly.

## 📈 Model Performance & Business Trade-offs
In e-commerce customer acquisition, **False Negatives (missing a digital shopper, losing a $150 sale) are far more expensive than False Positives (showing a digital ad to a physical shopper, costing $0.05).** Therefore, the final model was explicitly optimized for **Recall** rather than overall accuracy.


| Model Version | Accuracy | Digital Recall | Digital F1-Score | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (With Leakage)** | 95% | 0.87 | 0.87 | Biased by past spending data (Invalid). |
| **Multiclass Baseline (No Weights)**| 87% | 0.00 | 0.00 | Failed to detect minority digital classes. |
| **XGBoost Binary (Default 0.5 Threshold)** | 81% | 0.13 | 0.16 | Only caught 13% of actual digital shoppers. |
| **XGBoost Optimized (0.15 Threshold)** | **59%** | **0.55** | **0.26** | **Best for Business:** Increased minority-class detection by over 320% to drive customer acquisition. |


## 🛠️ Tech Stack
* **Language:** Python
* **Algorithms:** XGBoost, Random Forest
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
* **Techniques:** Threshold Tuning, Cost-Sensitive Learning, Precision-Recall Optimization
* **Environment:** Jupyter Notebook
