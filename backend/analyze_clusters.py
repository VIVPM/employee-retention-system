"""
Run this script AFTER training to analyze what each K-Means cluster represents.
Usage: cd backend && python analyze_clusters.py
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from kneed import KneeLocator

# 1. Load and preprocess (same steps as training pipeline)
df = pd.read_csv('../hr_employee_churn_data.csv')
df.drop('empid', axis=1, inplace=True)

# one-hot encode salary
cat_df = pd.get_dummies(df['salary'], prefix='salary', drop_first=True).astype(int)
df = pd.concat([df, cat_df], axis=1)
df.drop('salary', axis=1, inplace=True)

# handle missing values
if df.isnull().sum().any():
    imputer = KNNImputer(n_neighbors=3, weights='uniform')
    cols = df.columns
    df = pd.DataFrame(imputer.fit_transform(df), columns=cols)

# separate features and label
X = df.drop('left', axis=1)
y = df['left']

# 2. Find optimal clusters (elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
n_clusters = kn.knee
print(f"\nOptimal number of clusters: {n_clusters}\n")

# 3. Create clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Analyze each cluster
print("=" * 80)
print("CLUSTER PROFILE ANALYSIS")
print("=" * 80)

for c in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == c]
    total = len(cluster_data)
    left_count = cluster_data['left'].sum()
    left_pct = (left_count / total) * 100

    print(f"\n--- Cluster {c} ({total} employees, {left_pct:.1f}% churn rate) ---")
    print(f"  satisfaction_level    : {cluster_data['satisfaction_level'].mean():.2f}")
    print(f"  last_evaluation      : {cluster_data['last_evaluation'].mean():.2f}")
    print(f"  number_project       : {cluster_data['number_project'].mean():.1f}")
    print(f"  average_monthly_hours: {cluster_data['average_monthly_hours'].mean():.0f}")
    print(f"  time_spend_company   : {cluster_data['time_spend_company'].mean():.1f}")
    print(f"  Work_accident        : {cluster_data['Work_accident'].mean():.2f}")
    print(f"  promotion_last_5years: {cluster_data['promotion_last_5years'].mean():.2f}")
    print(f"  salary_low           : {cluster_data['salary_low'].mean():.2f}")
    print(f"  salary_medium        : {cluster_data['salary_medium'].mean():.2f}")

# 5. Summary comparison table
print("\n\n" + "=" * 80)
print("SUMMARY COMPARISON (mean values per cluster)")
print("=" * 80)
summary = df.groupby('Cluster').mean()
print(summary.to_string())

# 6. Cluster labels (auto-generated based on data)
print("\n\n" + "=" * 80)
print("CLUSTER INTERPRETATION")
print("=" * 80)
for c in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == c]
    avg = cluster_data.mean()
    overall = df.mean()

    traits = []
    if avg['satisfaction_level'] < overall['satisfaction_level'] - 0.1:
        traits.append("low satisfaction")
    elif avg['satisfaction_level'] > overall['satisfaction_level'] + 0.1:
        traits.append("high satisfaction")

    if avg['average_monthly_hours'] > overall['average_monthly_hours'] + 20:
        traits.append("overworked")
    elif avg['average_monthly_hours'] < overall['average_monthly_hours'] - 20:
        traits.append("low hours")

    if avg['number_project'] > overall['number_project'] + 0.5:
        traits.append("many projects")
    elif avg['number_project'] < overall['number_project'] - 0.5:
        traits.append("few projects")

    if avg['last_evaluation'] > overall['last_evaluation'] + 0.05:
        traits.append("high evaluation")
    elif avg['last_evaluation'] < overall['last_evaluation'] - 0.05:
        traits.append("low evaluation")

    if avg['time_spend_company'] > overall['time_spend_company'] + 0.5:
        traits.append("long tenure")
    elif avg['time_spend_company'] < overall['time_spend_company'] - 0.5:
        traits.append("short tenure")

    churn_pct = (cluster_data['left'].sum() / len(cluster_data)) * 100
    if churn_pct > 40:
        traits.append("HIGH CHURN RISK")
    elif churn_pct < 15:
        traits.append("low churn risk")

    label = ", ".join(traits) if traits else "average profile"
    print(f"  Cluster {c}: {label}")
