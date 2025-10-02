# 🚗 Vehicle Insurance Analysis  
**Data-Driven Risk Management and Customer Segmentation for Insurance Optimization**

## 📌 Project Overview
This project applies data analytics and machine learning techniques to optimize multiple aspects of the vehicle insurance industry, including:  
- Risk assessment and claims management  
- Customer segmentation and targeted pricing  
- Driver education initiatives  
- Fraud detection and security measures  

Vehicle insurance plays a key role in personal financial planning by protecting customers against accident, theft, and disaster-related losses. Insurance providers, however, face the challenge of balancing **profitability** with **competitive premiums**. This study leverages a large dataset containing information on **policies, customer demographics, vehicle attributes, and claims history** to explore these trade-offs.

---

## 📊 Dataset
- **Source:** `car_insurance.csv`  
- **Key Variables:**  
  - Customer demographics: age, driving experience  
  - Vehicle information: vehicle type, vehicle value  
  - Insurance details: premium, policy type  
  - Risk indicators: claims history, accidents  

### Exploratory Insights
- **Age Distribution:** Majority of policyholders are between 25–50 years old.  
- **Driving Experience:** Most customers have **0–9 years** of experience, making them higher risk.  
- **Vehicle Value:** Higher-value vehicles are associated with higher premiums and claim amounts.  
- **Claims History:** Customers with prior claims are significantly more likely to generate future claims.  

---

## 🔬 Methods
The following approaches were applied:  
- **Exploratory Data Analysis (EDA):** Histograms, scatter plots, correlation matrices  
- **Predictive Modeling:** Logistic Regression and Decision Trees to estimate claim probability  
- **Clustering (K-means):** Customer segmentation based on demographics, vehicle type, and risk profile  
- **Risk Modeling:** Incorporating claims history, driving experience, and vehicle value  

---

## 📈 Key Findings
1. **Young & Inexperienced Drivers → Higher Risk**  
   - Short driving experience strongly correlates with higher accident likelihood.  
   - Premium models should reflect the additional risk factor.  

2. **Claims History Matters**  
   - Past claims are a strong predictor of future claim probability.  
   - Useful for designing risk-adjusted premiums.  

3. **Vehicle Value Correlates with Premium & Risk**  
   - Higher-value cars incur higher claim amounts and require tailored insurance packages.  

4. **Customer Segmentation Enables Precision**  
   - Segmentation into **high-risk, medium-risk, and low-risk groups** improves pricing strategies and targeted education programs.  

---

## ✅ Conclusion
- **Data-driven risk modeling** enables insurers to identify high-risk customers and apply fair, competitive pricing.  
- **Customer segmentation** supports product differentiation and personalized insurance offerings.  
- **Driver education incentives** for young and inexperienced drivers can reduce accidents and improve loyalty.  

---

## 📊 Visualizations

### Age Distribution of Policyholders
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=data, x='age')
plt.title('Age Distribution of Policyholders')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```
### Driving Experience Distribution
```python
sns.histplot(data['driving_experience'], color='green')
plt.title('Distribution of Driving Experience (Years)')
plt.xlabel('Years of Driving Experience')
plt.ylabel('Count')
plt.show()
```
### Premium vs Vehicle Value
```python
sns.scatterplot(data=data, x='vehicle_value', y='premium', hue='claim')
plt.title('Premium vs Vehicle Value (with Claims)')
plt.xlabel('Vehicle Value')
plt.ylabel('Premium Amount')
plt.show()
```
### Customer Segmentation (K-Means Clustering)
```python
from sklearn.cluster import KMeans

X = data[['age', 'driving_experience', 'vehicle_value']]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
data['cluster'] = kmeans.labels_

sns.scatterplot(x=data['age'], y=data['vehicle_value'], hue=data['cluster'], palette='Set2')
plt.title('Customer Segmentation by Age & Vehicle Value')
plt.xlabel('Age')
plt.ylabel('Vehicle Value')
plt.show()
```
