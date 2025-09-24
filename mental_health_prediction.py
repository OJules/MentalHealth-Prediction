#!/usr/bin/env python
# coding: utf-8

# ## Depression Detection

# #### Data analysis

# In[12]:


# Imports nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


# In[13]:


# Chargeons les données
train = pd.read_csv('C:/Users/odjen/Downloads/train.csv')
test = pd.read_csv('C:/Users/odjen/Downloads/test (2).csv')


# In[16]:


train


# In[14]:


# 1. Analyse initiale des données manquantes
print("Nombre de valeurs manquantes par colonne:")
print(train.isnull().sum())


# In[15]:


# 2. Analyse des valeurs manquantes selon le statut
status_analysis = pd.pivot_table(
   train, 
   values=['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction'],
   columns='Working Professional or Student',
   aggfunc=lambda x: x.isna().sum()
)
print("\nValeurs manquantes par statut:")
print(status_analysis)


# In[17]:


# 3. Distribution de la variable cible
plt.figure(figsize=(8,5))
sns.countplot(data=train, x='Depression')
plt.title('Distribution de Depression')
plt.show()


# In[18]:


# 4. Corrélations avec la dépression
correlations = train.corr()['Depression'].sort_values()
plt.figure(figsize=(10,6))
correlations.plot(kind='bar')
plt.title('Corrélations avec Depression')
plt.xticks(rotation=45)
plt.show()


# In[20]:


# 5. Prétraitement des données
# Supprimer les NA dans certaines colonnes
train = train.dropna(subset=['Dietary Habits', 'Degree', 'Financial Stress'])

# Gestion des NA selon le statut
students = train[train['Working Professional or Student'] == 'Student'].copy()
professionals = train[train['Working Professional or Student'] == 'Professional'].copy()

# Pour étudiants
students.loc[:, 'Academic Pressure'] = students['Academic Pressure'].fillna(students['Academic Pressure'].mean())
students.loc[:, 'Study Satisfaction'] = students['Study Satisfaction'].fillna(students['Study Satisfaction'].mean())
students.loc[:, 'CGPA'] = students['CGPA'].fillna(students['CGPA'].mean())
students.loc[:, 'Work Pressure'] = 0
students.loc[:, 'Job Satisfaction'] = 0

# Pour professionnels
professionals.loc[:, 'Work Pressure'] = professionals['Work Pressure'].fillna(professionals['Work Pressure'].mean())
professionals.loc[:, 'Job Satisfaction'] = professionals['Job Satisfaction'].fillna(professionals['Job Satisfaction'].mean())
professionals.loc[:, 'Academic Pressure'] = 0
professionals.loc[:, 'Study Satisfaction'] = 0
professionals.loc[:, 'CGPA'] = 0

# Combiner
X_clean = pd.concat([students, professionals])


# In[21]:


# 6. Standardisation et préparation des features
# Modèle complet
scaler = StandardScaler()
numeric_cols = ['Financial Stress', 'Academic Pressure', 'Work Pressure', 'Work/Study Hours']
X = X_clean[numeric_cols].copy()
X = scaler.fit_transform(X)
y = X_clean['Depression']


# ### Modélisation 

# In[22]:


# 7. Modèles
# Régression logistique
lr = LogisticRegression(random_state=42)
lr.fit(X, y)
y_pred_lr = lr.predict(X)
print("\nClassification Report - Logistic Regression:")
print(classification_report(y, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y, y_pred_lr))


# In[23]:


# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
y_pred_rf = rf.predict(X)
print("\nClassification Report - Random Forest:")
print(classification_report(y, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y, y_pred_rf))


# In[24]:


# XGBoost
xgb = XGBClassifier(random_state=42)
xgb.fit(X, y)
y_pred_xgb = xgb.predict(X)
print("\nClassification Report - XGBoost:")
print(classification_report(y, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y, y_pred_xgb))


# In[25]:


# 8. Modèle simplifié (2 features)
X_simple = X_clean[['Academic Pressure', 'Financial Stress']].copy()
X_simple = scaler.fit_transform(X_simple)

models = {
   'Logistic Regression': LogisticRegression(random_state=42),
   'Random Forest': RandomForestClassifier(random_state=42),
   'XGBoost': XGBClassifier(random_state=42)
}

for name, model in models.items():
   model.fit(X_simple, y)
   y_pred = model.predict(X_simple)
   print(f"\n{name}:")
   print(classification_report(y, y_pred))
   print(f"ROC-AUC Score: {roc_auc_score(y, y_pred)}")


# ### Testons les modèles 

# In[2]:


test


# ### prétraitement du test set 

# In[27]:


# Prétraitement test set
# 1. Gérons les NA selon statut
test_students = test[test['Working Professional or Student'] == 'Student'].copy()
test_professionals = test[test['Working Professional or Student'] == 'Professional'].copy()

# Étudiants
test_students.loc[:, 'Academic Pressure'] = test_students['Academic Pressure'].fillna(test_students['Academic Pressure'].mean())
test_students.loc[:, 'Study Satisfaction'] = test_students['Study Satisfaction'].fillna(test_students['Study Satisfaction'].mean())
test_students.loc[:, 'CGPA'] = test_students['CGPA'].fillna(test_students['CGPA'].mean())
test_students.loc[:, 'Work Pressure'] = 0
test_students.loc[:, 'Job Satisfaction'] = 0

# Professionnels
test_professionals.loc[:, 'Work Pressure'] = test_professionals['Work Pressure'].fillna(test_professionals['Work Pressure'].mean())
test_professionals.loc[:, 'Job Satisfaction'] = test_professionals['Job Satisfaction'].fillna(test_professionals['Job Satisfaction'].mean())
test_professionals.loc[:, 'Academic Pressure'] = 0
test_professionals.loc[:, 'Study Satisfaction'] = 0
test_professionals.loc[:, 'CGPA'] = 0

# Combinons les subsets
X_test_clean = pd.concat([test_students, test_professionals])


# In[28]:


# Standardisation
# Pour modèle complet (4 features)
scaler_complet = StandardScaler()
X = X_clean[numeric_cols].copy()
X = scaler_complet.fit_transform(X)

# Réentraînement des modèles
models_complet = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

for name, model in models_complet.items():
    model.fit(X, y)

# Transformation du test set
X_test = X_test_clean[numeric_cols].copy()
X_test = scaler_complet.transform(X_test)

# Prédictions
for name, model in models_complet.items():
    y_pred = model.predict(X_test)
    print(f"\nPrédictions {name}:")
    print(pd.Series(y_pred).value_counts())


# ### Comparaison des modèles 

# In[29]:


# Distribution des prédictions par modèle
plt.figure(figsize=(15,5))

for i, (name, model) in enumerate(models_complet.items(), 1):
   y_pred = model.predict(X_test)
   plt.subplot(1,3,i)
   sns.countplot(data=pd.DataFrame({'predictions': y_pred}), x='predictions')
   plt.title(f'Distribution - {name}')

plt.tight_layout()
plt.show()


# ###  Choix du modèle

# In[34]:


# Sélection du Random Forest
best_model = models_complet['Random Forest']


# ### Deploiement 

# In[35]:


# Sauvegarde du modèle et du scaler
joblib.dump(best_model, 'depression_rf_model.pkl')
joblib.dump(scaler_complet, 'depression_scaler.pkl')


# In[36]:


# Fonction de prédiction
def predict_depression(data):
   """
   Prédit la dépression à partir des features: 
   Financial Stress, Academic Pressure, Work Pressure, Work/Study Hours
   """
   scaled_features = scaler_complet.transform(data)
   prediction = best_model.predict(scaled_features)
   return prediction


# In[37]:


# Test de la fonction
test_data = X_test[:1]  # Premier exemple du test set
print("Prédiction:", predict_depression(test_data))

