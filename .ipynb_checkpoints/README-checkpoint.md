# Ecommerce Customer Behaviour Analysis & Recommendation System

## Project Overview

This project analyzes customer purchasing behaviour in an ecommerce platform using **unsupervised machine learning** techniques.

The project performs:

* Customer segmentation using clustering algorithms
* Product recommendation using association rule mining
* Data preprocessing and feature engineering
* Interactive dashboard using Streamlit

---

# Objectives

1. Analyze customer purchasing patterns
2. Segment customers into different behavioural groups
3. Discover product associations for recommendations
4. Deploy results using a Streamlit web application

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Mlxtend
* Matplotlib
* Seaborn
* Streamlit

---

# Dataset Features

The dataset contains ecommerce transaction data including:

* CustomerID
* Product
* Category
* Price
* Quantity
* PaymentType
* SessionDuration
* Rating
* DiscountApplied
* Device
* ShippingType
* ReviewText

---

# Data Processing Steps

1. Data Collection
2. Data Cleaning

   * Handling Missing Values
   * Removing Duplicate Rows
   * Fixing Inconsistent Data
3. Data Wrangling
4. Exploratory Data Analysis
5. Feature Engineering

---

# Machine Learning Models Used

## Customer Segmentation

Clustering Algorithms:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Model

These algorithms group customers based on purchasing behaviour.

---

# Recommendation System

Association rule mining is used to identify product relationships.

Example Rules:

* Mobile → Earphones
* Laptop → Mouse
* Dress → Heels

Algorithms used:

* Apriori
* FP-Growth
* ECLAT

---

# Project Structure

```
ecommerce-customer-behaviour-analysis

data/
notebooks/
models/
association_rules/
results/
app/
requirements.txt
README.md
```

---

# Streamlit Dashboard

The Streamlit app displays:

* Customer segments
* Cluster insights
* Product recommendation rules

Run the dashboard using:

```
streamlit run app/streamlit_app.py
```

---

# Future Improvements

* Deep learning based recommendation systems
* Real-time recommendation engine
* Advanced customer lifetime value prediction

---

# Author

Keertiraj Kamble
Artificial Intelligence & Data Science Student
