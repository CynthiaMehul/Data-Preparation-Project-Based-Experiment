# Project Based Experiment - Data Preparation Techniques
# Name: Cynthia Mehul J
# Reg No.: 212223240020
## Aim:
To perform various data preparation techniques and visualize the data distribution using different plots for analysing the credit card transaction dataset.

## Dataset:
Credit card transaction dataset with 10,000 sampled records.

## Program:

### Import required libraries.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
```

### Read the csv file and display the first five rows.
```python
df = pd.read_csv("C:/Users/admin/Documents/AIML/SEM 5/ds/project/creditcard.csv").sample(10000, random_state=42)
df.head()
```

### Check for number of null values.
```python
df.isnull().sum()
```

### Outlier Detection
```python
sns.boxplot(df['Amount'])
plt.title("Outliers in 'Amount'")
plt.show()

z_scores = np.abs(stats.zscore(df['Amount']))
outliers = df[z_scores > 3]
print("Number of outliers:", len(outliers))
```

### Remove Outliers
```python
df_clean = df[z_scores <= 3]
df_clean.shape
```

### Log Transformation
```python
df_clean['Amount_log'] = np.log1p(df_clean['Amount'])
sns.histplot(df_clean['Amount_log'], kde=True)
plt.title("Log-Transformed Amount")
plt.show()
```

### Standard Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean['Amount_scaled'] = scaler.fit_transform(df_clean[['Amount']])
df_clean[['Amount', 'Amount_scaled']].head()
```

### Bar Plot: Class vs Amount
```python
sns.barplot(x='Class', y='Amount', data=df_clean)
plt.title("Average Transaction Amount by Class")
plt.show()
```

### High vs Low Amount Boxplot
```python
df_clean['HighAmount'] = df_clean['Amount'] > df_clean['Amount'].median()
sns.boxplot(x='HighAmount', y='Amount', data=df_clean)
plt.title("Amount Distribution: High vs Low Amount")
plt.show()
```

### Amount vs Class Scatter
```python
sns.scatterplot(x='Amount', y='Class', data=df_clean)
plt.title("Fraud Occurrence Based on Amount")
plt.show()
```

### Amount Distribution by Class
```python
sns.boxplot(x='Class', y='Amount', data=df_clean)
plt.title("Amount Distribution by Class")
plt.show()
```

### Amount vs Time with Fraud Highlighted
```python
sns.scatterplot(x='Time', y='Amount', hue='Class', data=df_clean)
plt.title("Amount vs Time (With Fraud Highlighted)")
plt.show()
```

### Amount by Time Segments
```python
df_clean['TimeSegment'] = pd.qcut(df_clean['Time'], 4, labels=["Morning","Afternoon","Evening","Night"])

sns.histplot(data=df_clean, x='Amount', hue='TimeSegment', kde=True)
plt.title("Distribution of Amount Across Time Segments")
plt.show()
```

### Average Amount by Amount Bins
```python
df_clean['AmountBin'] = pd.qcut(df_clean['Amount'], 5)

group_avg = df_clean.groupby('AmountBin')['Amount'].mean()

plt.plot(group_avg.index.astype(str), group_avg.values)
plt.xticks(rotation=45)
plt.title("Average Amount by Amount Group")
plt.show()
```

### Boxplot per Class Label
```python
sns.boxplot(x='Class', y='Amount', data=df_clean)
plt.title("Amount Distribution per Class Label")
plt.show()
```

### Violin Plot of Amount by Time Segment
```python
sns.violinplot(x='TimeSegment', y='Amount', data=df_clean)
plt.title("Transaction Amount by Time Segment")
plt.show()
```

### Scatter Plot: Amount vs Time
```python
sns.scatterplot(x='Amount', y='Time', data=df_clean)
plt.title("Correlation: Amount vs Time")
plt.show()
```


## Inference

1. **Outliers in Amount:**  
   The amount field contains many extreme outliers, indicating highly skewed transaction values.

<img width="762" height="584" alt="image" src="https://github.com/user-attachments/assets/12e08bd6-dc98-4238-9c17-be1fcc303323" />


2. **Outlier Removal:**  
   Removing values with z-score > 3 reduces noise and improves data consistency.

<img width="318" height="100" alt="image" src="https://github.com/user-attachments/assets/04b09c1a-cd31-49ab-96d0-729956094a14" />

3. **Log Transformation:**  
   Log-transform normalizes the distribution, reducing skewness and making the data suitable for modeling.

<img width="770" height="571" alt="image" src="https://github.com/user-attachments/assets/533f35cd-14ca-402e-a6e3-3fae0605cc84" />

4. **Feature Scaling:**  
   StandardScaler standardizes the “Amount” variable for machine learning algorithms.

<img width="300" height="219" alt="image" src="https://github.com/user-attachments/assets/5f1bffae-fbd4-43b3-89f9-e9362833d584" />

5. **Average Amount by Class:**  
   Fraud cases (Class 1) show slightly higher mean transaction amounts.

<img width="760" height="572" alt="image" src="https://github.com/user-attachments/assets/6bc4a267-4bae-486d-b562-8a3f2fac2ef3" />

6. **High vs Low Amount:**  
   High-value transactions exhibit more variability and more extreme values than low-value ones.

<img width="769" height="574" alt="image" src="https://github.com/user-attachments/assets/32721f35-fbab-4408-aec0-8f8f687a6418" />

7. **Fraud vs Amount Scatter:**  
   Fraud occurs at all transaction amounts, so amount alone cannot distinguish fraud.

<img width="755" height="565" alt="image" src="https://github.com/user-attachments/assets/c6404ee0-c3fe-4adb-a6e2-8266dd6ada15" />

8. **Amount by Class:**  
   Fraud shows more spread but both classes overlap, it’s not a reliable predictor.

<img width="773" height="574" alt="image" src="https://github.com/user-attachments/assets/f938ab6b-b58a-4805-a4d0-041aa73f3fca" />

9. **Amount vs Time:**  
   Fraud cases rarely occur in the given dataset sample.

<img width="764" height="567" alt="image" src="https://github.com/user-attachments/assets/724305fa-f4dc-4c33-8187-6712448bfd1c" />

10. **Amount by Time Segment:**  
    Transaction amounts are high in evening time.

<img width="783" height="565" alt="image" src="https://github.com/user-attachments/assets/65c26c31-be3e-4fcc-b72f-fb1a8422b267" />

11. **Average Amount by Bins:**  
    The trend confirms the natural skewness of transaction values.

<img width="785" height="645" alt="image" src="https://github.com/user-attachments/assets/1a34736f-650f-4a77-bc70-ccdb8a32a0c9" />

12. **Violin Plot:**  
    Evening and Night periods show more variation in amounts.

<img width="769" height="567" alt="image" src="https://github.com/user-attachments/assets/8591fbf2-6169-4740-92ac-17edf04b31f6" />

13. **Amount vs Time (Overall):**  
    No strong correlation between amount and time, values appear randomly spread.

<img width="802" height="564" alt="image" src="https://github.com/user-attachments/assets/13bc26c3-78dc-48cb-af63-c1b5b7f9bec0" />

## Result
Therefore, various data preparation techniques such as outlier detection, log transformation, scaling, segmentation, and visualization were performed successfully.

