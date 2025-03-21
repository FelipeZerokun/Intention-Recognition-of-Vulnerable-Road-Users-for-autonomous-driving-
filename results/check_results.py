import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

csv_file = 'test_predictions.csv'

list_predictions = pd.read_csv(csv_file)

labels = 'Standing Still', 'Walking'
cm = confusion_matrix(list_predictions['Ground Truth'], list_predictions['Predictions'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Action Recognition')
plt.show()
