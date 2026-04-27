# 🛒 Ecommerce Customer Behaviour Analysis & Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Mlxtend](https://img.shields.io/badge/Mlxtend-Association%20Rules-4C8CBF?style=flat-square)](http://rasbt.github.io/mlxtend/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

> **Unsupervised Machine Learning pipeline** that segments ecommerce customers into behavioural clusters and generates intelligent product recommendations using association rule mining — deployed as a live Streamlit web application.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Highlights](#-key-highlights)
- [Tech Stack](#-tech-stack)
- [Dataset Description](#-dataset-description)
- [Project Architecture](#-project-architecture)
- [Methodology](#-methodology)
- [Machine Learning Models](#-machine-learning-models)
- [Recommendation System](#-recommendation-system)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Project Overview

This end-to-end data science project analyzes real-world ecommerce transaction data to uncover hidden customer patterns and generate actionable product recommendations. The system applies **unsupervised machine learning** — no labelled data required — making it scalable and domain-agnostic.

**Business Problem Solved:**
- Who are my most valuable customer segments?
- Which products are frequently bought together?
- How can I personalize recommendations at scale?

**Solution Delivered:**
- Multi-algorithm customer segmentation with behavioral profiling
- Association rule mining for smart product bundling and upsell recommendations
- Interactive Streamlit dashboard for business stakeholders

---

## ✨ Key Highlights

| Area | Detail |
|---|---|
| **Domain** | Ecommerce / Retail Analytics |
| **ML Type** | Unsupervised Learning |
| **Segmentation Algorithms** | K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Model |
| **Recommendation Algorithms** | Apriori, FP-Growth, ECLAT |
| **Deployment** | Streamlit Web Application |
| **Dataset Features** | 12 transaction-level features |
| **Use Case** | Customer Intelligence + Product Recommendation |

---

## 🛠 Tech Stack

### Core Languages & Libraries

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Mlxtend |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Notebook Environment** | Jupyter Notebook |

### ML Algorithms Used

```
Clustering          : K-Means | Hierarchical | DBSCAN | Gaussian Mixture Model
Association Mining  : Apriori | FP-Growth | ECLAT
Evaluation Metrics  : Silhouette Score | Davies-Bouldin Index | Inertia
```

---

## 📊 Dataset Description

The dataset contains ecommerce transaction records with the following features:

| Feature | Type | Description |
|---|---|---|
| `CustomerID` | Categorical | Unique identifier per customer |
| `Product` | Categorical | Name of purchased product |
| `Category` | Categorical | Product category (Electronics, Fashion, etc.) |
| `Price` | Numerical | Transaction price |
| `Quantity` | Numerical | Number of units purchased |
| `PaymentType` | Categorical | Payment method used |
| `SessionDuration` | Numerical | Time spent on platform (seconds) |
| `Rating` | Numerical | Customer product rating (1–5) |
| `DiscountApplied` | Boolean | Whether a discount was used |
| `Device` | Categorical | Device type (Mobile, Desktop, Tablet) |
| `ShippingType` | Categorical | Shipping preference |
| `ReviewText` | Text | Customer review (NLP-ready) |

---

## 🏗 Project Architecture

```
Raw Transaction Data
        │
        ▼
┌─────────────────────────────┐
│     Data Preprocessing      │
│  (Cleaning, Wrangling, EDA) │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐   ┌──────────────────┐
│  Feature    │   │  Market Basket   │
│ Engineering │   │  Transformation  │
└──────┬──────┘   └────────┬─────────┘
       ▼                   ▼
┌─────────────┐   ┌──────────────────┐
│  Clustering │   │  Association Rule│
│  Algorithms │   │  Mining          │
└──────┬──────┘   └────────┬─────────┘
       ▼                   ▼
┌──────────────────────────────────┐
│     Streamlit Dashboard          │
│  (Segments + Recommendations)    │
└──────────────────────────────────┘
```

---

## 🔬 Methodology

### Phase 1 — Data Collection & Cleaning
- Loaded raw ecommerce transaction CSV data
- Handled missing values using median/mode imputation strategies
- Removed duplicate transaction records
- Fixed inconsistent categorical entries (e.g., normalizing product names, device labels)

### Phase 2 — Exploratory Data Analysis (EDA)
- Analyzed purchase frequency distributions across categories
- Identified top-selling products, peak session times, and device usage patterns
- Visualized rating distributions, discount impact, and payment type preferences
- Correlation heatmaps to understand feature relationships

### Phase 3 — Feature Engineering
- Aggregated transaction-level data to customer-level RFM-style features
- Engineered: total spend, purchase frequency, average order value, preferred category
- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Normalized numerical features using StandardScaler for clustering

### Phase 4 — Model Training & Evaluation
- Ran multiple clustering algorithms and compared using Silhouette Score
- Used the Elbow Method and Dendrogram analysis for optimal cluster selection
- Mined association rules with configurable support and confidence thresholds

---

## 🤖 Machine Learning Models

### Customer Segmentation

Four clustering algorithms were benchmarked to identify the most stable customer groupings:

#### K-Means Clustering
- Partitions customers into `k` clusters by minimizing intra-cluster variance
- Optimal `k` determined using the **Elbow Method** and **Silhouette Score**

#### Hierarchical Clustering
- Builds a dendrogram to reveal nested customer groupings
- Agglomerative approach with Ward linkage for compact, well-separated clusters

#### DBSCAN (Density-Based Spatial Clustering)
- Identifies clusters of arbitrary shape
- Automatically detects outlier/noise customers who don't fit standard segments
- Useful for spotting anomalous purchasing behaviour

#### Gaussian Mixture Model (GMM)
- Probabilistic soft-assignment of customers to clusters
- Handles overlapping segments where customers exhibit mixed behaviour patterns

**Evaluation Metrics:**
```
Silhouette Score     → Measures cluster cohesion and separation (higher = better)
Davies-Bouldin Index → Measures intra-cluster similarity (lower = better)
Inertia (K-Means)    → Within-cluster sum of squared distances
```

---

## 🔗 Recommendation System

Association Rule Mining extracts frequent itemsets to discover which products customers buy together.

### Example Rules Discovered

| If Customer Buys → | Then Recommend → | Confidence |
|---|---|---|
| Mobile | Earphones | High |
| Laptop | Mouse | High |
| Dress | Heels | Medium |
| Camera | Memory Card | High |
| Tablet | Keyboard Cover | Medium |

### Algorithms Used

| Algorithm | Approach | Strength |
|---|---|---|
| **Apriori** | Candidate generation + pruning | Simple, interpretable |
| **FP-Growth** | Frequent Pattern tree | Faster, no candidate generation |
| **ECLAT** | Vertical data format + intersection | Memory efficient for dense data |

**Key Metrics:**
- **Support** — How often the itemset appears in transactions
- **Confidence** — How often the rule is correct
- **Lift** — How much more likely the association is vs. random chance (Lift > 1 = meaningful)

---

## 📱 Streamlit Dashboard

The interactive dashboard provides a business-ready view of the analysis:

**Dashboard Sections:**
- **Customer Segments View** — Scatter plots of clusters with behavioral profiles
- **Cluster Insights** — Summary statistics per segment (avg spend, frequency, preferred category)
- **Association Rules Explorer** — Filterable rules by support, confidence, and lift thresholds
- **Product Recommender** — Input a product → get top associated items instantly

**Run Locally:**

```bash
streamlit run app/streamlit_app.py
```

Then open your browser at `http://localhost:8501`

---

## ⚙ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Keertiraj2004/ecommerce-customer-behaviour-analysis.git
cd ecommerce-customer-behaviour-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter notebooks (for exploration)
jupyter notebook notebooks/

# 5. Run the Streamlit dashboard
streamlit run app/streamlit_app.py
```

### Dependencies (`requirements.txt`)

```
pandas
numpy
scikit-learn
mlxtend
matplotlib
seaborn
streamlit
jupyter
```

---

## 📁 Project Structure

```
ecommerce-customer-behaviour-analysis/
│
├── data/                          # Raw and processed datasets
│   └── ecommerce_data.csv
│
├── notebooks/                     # Jupyter notebooks for EDA and modeling
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_customer_segmentation.ipynb
│   └── 04_association_rule_mining.ipynb
│
├── models/                        # Saved clustering model objects
│   └── kmeans_model.pkl
│
├── association_rules/             # Exported rule sets (CSV/JSON)
│   └── rules_apriori.csv
│
├── results/                       # Cluster plots, visualizations, reports
│   ├── cluster_plots/
│   └── eda_plots/
│
├── app/                           # Streamlit dashboard application
│   └── streamlit_app.py
│
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation (this file)
```

---

## 📈 Results

**Customer Segmentation:**
- Successfully identified distinct customer behavioural groups
- Segments include high-value frequent buyers, discount-driven shoppers, casual browsers, and high-rating loyal customers
- DBSCAN isolated a small percentage of anomalous/outlier customers

**Recommendation System:**
- Generated high-confidence product association rules with Lift > 1.5
- Cross-category rules (e.g., Electronics → Accessories) provide actionable upsell opportunities
- Rules validated against held-out transaction subsets

---

## 🚀 Future Improvements

- [ ] **Deep learning recommendations** — Implement Neural Collaborative Filtering (NCF) or Autoencoders for personalized recommendations
- [ ] **Real-time recommendation engine** — Integrate with a streaming pipeline (Apache Kafka + Redis) for live inference
- [ ] **Customer Lifetime Value (CLV) prediction** — Regression models to forecast future customer revenue
- [ ] **NLP on ReviewText** — Sentiment analysis and topic modelling on review data
- [ ] **Cloud deployment** — Deploy Streamlit app on Streamlit Cloud, AWS, or GCP
- [ ] **A/B testing framework** — Measure recommendation impact on conversion rate

---

## 👤 Author

**Keertiraj Kamble**
B.E. in Artificial Intelligence & Data Science
KLE College of Engineering and Technology (VTU), Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-Keertiraj2004-181717?style=flat-square&logo=github)](https://github.com/Keertiraj2004)
[![Email](https://img.shields.io/badge/Email-keertirajkamble023%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:keertirajkamble023@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project useful, please consider giving it a star on GitHub — it helps others discover it!
