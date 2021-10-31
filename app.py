#!/usr/bin/env python
# coding: utf-8

# # Data Mining Project
# By:
# - Yap Wei Lun 1171103109
# - Ng Ruiwen 1171103253
# - Teo Yan Xue 1171103499

# In[130]:

import streamlit as st
from PIL import Image 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[131]:


df = pd.read_csv("LaundryData.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)
Image.open('mmu_logo.png').convert('RGB').save('mmu_logo.png')
im = Image.open("mmu_logo.png")
st.image(im, width=300)

st.title("DATA MINING PROJECT")
st.markdown("By:")
st.markdown(" - Yap Wei Lun 1171103109")
st.markdown("- Ng Ruiwen 1171103253") 
st.markdown(" - Teo Yan Xue 1171103499")
st.header("Part 1: Exploratory Data Analysis and Data Pre-Processing")
st.subheader("Data Set")
df

# # Exploratory Data Analysis

# ### Q1: Does body size have relationship with basket size?
# 
# Answer: Based on the graph, majority of the customers have big basket size regardless of their body size. However, there are still chances of thin customers bringing small basket to the laundry shop.

# In[132]:


st.subheader("Q1: Does body size have relationship with basket size?")
pd.crosstab(df.Basket_Size,df.Body_Size).plot(kind='bar')
st.pyplot()
st.markdown("Answer: Based on the graph, majority of the customers have big basket size regardless of their body size. However, there are still chances of thin customers bringing small basket to the laundry shop.")


# ### Q2: What kind of attire do the customers wear at specific hours?
# 
# Answer: It is shown that at 4am, a lot of customers wearing casual and formal attire will come to the laundry shop, while customers that wear traditional attire will usually come at 4am, 11am, and 5pm.

# In[133]:

st.subheader("Q2: What kind of attire do the customers wear at specific hours?")
df['datetime'] = pd.to_datetime(df['Time'])
df['hour'] = df['datetime'].dt.hour
pd.crosstab(df.Attire,df.hour).plot(kind='bar', figsize=(10,10))
st.pyplot()
st.markdown("Answer: It is shown that at 4am, a lot of customers wearing casual and formal attire will come to the laundry shop, while customers that wear traditional attire will usually come at 4am, 11am, and 5pm.")

# ### Q3: Which date in the dataset has the most/least customers?
# 
# Answer: November 29, 2015 has the most customers (134). October 30, 2015 has the least customer (1).

# In[134]:


st.subheader("Q3: Which date in the dataset has the most/least customers?")
#df_groupby_date = df.groupby('Date')
#df_groupby_date.size()

df_pplCount = pd.read_csv("LaundryData.csv")
df_pplCount['People_Count'] = 1
df_pplCount = df_pplCount.groupby('Date', as_index=False)['People_Count'].sum()
df_pplCount
st.markdown("Answer: November 29, 2015 has the most customers (134). October 30, 2015 has the least customer (1).")


# ### Q4: What kind of items do customers wash at different age range?
# 
# Answer: Regardless of the age range, most of the customers wash clothes more than blankets.

# In[135]:


st.subheader("Q4: What kind of items do customers wash at different age range?")
analysis = df.copy()

analysis['AGE_GROUP'] = pd.cut(analysis['Age_Range'],[20,30,40,50,60,70])
analysis['AGE_GROUP'].value_counts()


# In[136]:


pd.crosstab(df.Wash_Item,analysis['AGE_GROUP']).plot(kind='bar')
st.pyplot()
st.markdown("Answer: Regardless of the age range, most of the customers wash clothes more than blankets.")


# ### Q5: Does the basket size determine which washer and dryer to use?
# 
# Answer: From the graph below, we can assume that washer no 3 and no 6 are large washing machines, washer no 4 is a medium sized washing machine and washer no 3 is a small sized washing machine. We can also assume that dryer no 7 can allocate large basket size worth of clothes.

# In[137]:


st.subheader("Q5: Does the basket size determine which washer and dryer to use?")
pd.crosstab(df.Basket_Size,df.Washer_No).plot(kind='bar')


# In[138]:


pd.crosstab(df.Basket_Size,df.Dryer_No).plot(kind='bar')
st.pyplot()
st.markdown("Answer: From the graph below, we can assume that washer no 3 and no 6 are large washing machines, washer no 4 is a medium sized washing machine and washer no 3 is a small sized washing machine. We can also assume that dryer no 7 can allocate large basket size worth of clothes.")


# ### Q6: Which gender and race of the customer comes the most?
# 
# Answer: Female Malays come the most. In terms of male, the Indians come the most.

# In[139]:


st.subheader("Q6: Which gender and race of the customer comes the most?")
df["Gender"].fillna("Unknown", inplace = True)
df["Gender"].value_counts()
df["Race"].fillna("others", inplace = True)
df["Race"].value_counts()
pd.crosstab(df.Race,df.Gender).plot(kind='bar')
st.pyplot()
st.markdown("Answer: Female Malays come the most. In terms of male, the Indians come the most.")


# # Data Pre-Processing

# ## At this part, we removed data that we will not be using anymore, and filled up some empty data.
# 
# In this case, we removed "No", "Date", and "Time". Then, we fill up the empty data under "With_Kids" and "Kids_Category" which if With_Kids is 'no', then the empty data in Kids_Category will be 'no_kids'. If the data in "With_Kids" is 'yes', then the "Kids_Category" will be filled with 'unknown', because we cannot determine the exact category of the kid, but we do know the customer is with a/some kid/s. 

# In[140]:


st.subheader("Data Pre-Processing")
st.markdown("At this part, we removed data that we will not be using anymore, and filled up some empty data.")
st.markdown("In this case, we removed 'No', 'Date', and 'Time'. Then, we fill up the empty data under 'With_Kids' and 'Kids_Category' which if With_Kids is 'no', then the empty data in Kids_Category will be 'no_kids'. If the data in 'With_Kids' is 'yes', then the 'Kids_Category' will be filled with 'unknown', because we cannot determine the exact category of the kid, but we do know the customer has kid(s). ")
st.markdown("Null race values are also replaced with 'others' under the assumption that they are within the 3 main races in Malaysia but not under the category of foreigners")
st.markdown("Null gender values are replaced with 'unknown' as there might be those that choses not to reveal their gender or look androgynous")
df = df.drop(columns=['No'])
df = df.drop(columns=['Date', 'Time'])

m = df['With_Kids'].isna()
df.loc[m, 'With_Kids'] = np.where(df.loc[m, 'Kids_Category'].eq('no_kids') & df.loc[m, 'Kids_Category'].ne(None), 'no', None)
n = df['With_Kids'].isna()
df.loc[n, 'With_Kids'] = np.where(df.loc[n, 'Kids_Category'].isna(), None, 'yes')

m = df['Kids_Category'].isna()
df.loc[m, 'Kids_Category'] = np.where(df.loc[m, 'With_Kids'].eq('no'), 'no_kids', None)
n = df['Kids_Category'].isna()
df.loc[n, 'Kids_Category'] = np.where(df.loc[n, 'With_Kids'].isna(), None, 'unknown')
df

# ## Association Rule Mining

# We decided to perform associate rule mining on the washer no and the dryer no to see which one is tend to be used next if one washer/dryer is chosen.

# In[141]:


st.subheader("Association Rule Mining")
st.markdown("At this part, we removed data that we will not be using anymore, and filled up some empty data.")
from apyori import apriori
df_ARM = df[['Washer_No', 'Dryer_No']]
records = []
for i in range(0, 807):
    records.append([str(df_ARM.values[i,j]) for j in range(0, 2)])


# From here, we can see that if washer no 3 is chosen, dryer no 7 will be used next. Then washer 4 -> dryer 8, washer 5 -> dryer 9, washer 6 -> dryer 10.

# In[142]:


st.markdown(" - From here, we can see that if washer no 3 is chosen, dryer no 7 will be used next. Then washer 4 -> dryer 8, washer 5 -> dryer 9, washer 6 -> dryer 10.")
st.markdown(" - Therefore, we assume that when we run ARM, the results would be very similar.")
grouped_df = df.groupby(["Washer_No", "Dryer_No"], as_index=False)
grouped_df = grouped_df.size()
grouped_df


# In[143]:


association_rules = apriori(records, min_support=0.05, min_confidence=0.2, min_lift=1.3, min_length=2)
association_results = list(association_rules)
#association_results


# In[144]:


cnt =0

for item in association_results:
    cnt += 1
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    st.markdown("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])

    #second index of the inner list
    st.markdown("Support: " + str(round(item[1],3)))

    #third index of the list located at 0th
    #of the third index of the inner list

    st.markdown("Confidence: " + str(round(item[2][0][2],4)))
    st.markdown("Lift: " + str(round(item[2][0][3],4)))
    st.markdown("=====================================")


# # Feature Selection

# Here, we use the boruta package and RandomForestClassifier to perform our feature selection.

# In[145]:

st.header("Part 2: Feature Selection")
st.subheader("Preparation")
st.markdown("Here, we use the boruta package and RandomForestClassifier to perform our feature selection.")
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# In[146]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# ## Label Encoding
# 
# - We made a copy of our dataframe and use 'Wash_Item' as our independent variable. We also dropped datas that have NaN value inside to help increase the quality of the data since we cannot fill up most of them manually.
# - From here, our total data in this dataset reduced from 807 to 684.

# In[147]:


st.subheader("Label Encoding")
st.markdown("We made a copy of our dataframe and use 'Wash_Item' as our independent variable. We also dropped datas that have NaN value inside to help increase the quality of the data since we cannot fill up most of them manually.")
st.markdown("From here, our total data in this dataset reduced from 807 to 684.")
df_FS = df.copy()
df_FS['Wash_Item'].unique()
df_FS = df_FS.dropna()
df_FS = df_FS.drop(['datetime', 'hour'], axis='columns')

le = LabelEncoder()

for col in df_FS.columns:
    if df_FS[col].dtypes == object:
        df_FS[col] = le.fit_transform(df_FS[col])

y = df_FS['Wash_Item']
X = df_FS.drop('Wash_Item', axis='columns')
colnames = X.columns
df_FS


# In[148]:


st.subheader("Heatmap")
st.markdown("To check if there's any correlations between the datas in the dataset")
corrMatrix2 = df_FS.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corrMatrix2, vmax=.8, square=True, annot=True, fmt= '.2f', annot_kws={'size': 15}, cmap=sns.color_palette("Reds"))
st.pyplot()
st.markdown("Answer: According to the heatmap, With_Kids has high correlation with Kids_Category. Another interesting find is the correlation between gender and shirt_type.")


st.subheader("Feature Selection")
st.markdown("Here, we perform our feature selection using boruta and RFC.")

rf = RandomForestClassifier(n_jobs=-1,class_weight="balanced",max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state =1)

feat_selector.fit(X.values,y.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score",ascending=False)


# In[149]:


st.markdown('---------Top 5----------')
a = boruta_score.head(5)
a

st.markdown('---------Bottom 5----------')
a = boruta_score.tail(5)
a

# In[150]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:25], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("Boruta all Features")
st.pyplot()

# In[151]:




# In[152]:


st.header("Part 3: Machine Learning Techniques")
st.markdown("Will discuss about classification, regression and clustering.")
st.header("Classification")
st.markdown("Here, we use Naive Bayes Classifier, Random Forest Classifier, and KNN Classifier to compare which model is the best to use.")
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[153]:


test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=10)


# In[154]:


st.subheader("Naive Bayes Classifier")
nb = GaussianNB()
nb.fit(X_train, y_train)

# Calculate the overall accuracy on test set 

st.markdown("Accuracy on test set: {:.3f}".format(nb.score(X_test, y_test)))

# Calculate AUC
prob_nb = nb.predict_proba(X_test)
prob_nb = prob_nb[:, 1]

auc_nb = roc_auc_score(y_test, prob_nb)

st.markdown("AUC for NB: {:.2f}".format(auc_nb))

# your codes here...
y_pred = nb.predict(X_test)
confusion_majority=confusion_matrix(y_test, y_pred)

st.markdown('Majority classifier Confusion Matrix\n')
confusion_majority

st.markdown('**')
st.markdown('Majority TN= ' + confusion_majority[0][0].astype(str))
st.markdown('Majority FP= ' + confusion_majority[0][1].astype(str))
st.markdown('Majority FN= ' + confusion_majority[1][0].astype(str))
st.markdown('Majority TP= ' + confusion_majority[1][1].astype(str))
st.markdown('**')

st.markdown('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.markdown('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.markdown('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.markdown('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


# In[155]:


st.subheader("Random Forest Classifier")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
 
# Calculate the overall accuracy on test set 

st.markdown("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

# Calculate AUC
# your codes here...
prob_rf = rf.predict_proba(X_test)
prob_rf = prob_rf[:, 1]

auc_rf = roc_auc_score(y_test, prob_rf)

st.markdown("AUC for RF: {:.2f}".format(auc_rf))

y_pred = rf.predict(X_test)
confusion_majority=confusion_matrix(y_test, y_pred)

st.markdown('Majority classifier Confusion Matrix\n')
confusion_majority

st.markdown('**')
st.markdown('Majority TN= ' + confusion_majority[0][0].astype(str))
st.markdown('Majority FP= ' + confusion_majority[0][1].astype(str))
st.markdown('Majority FN= ' + confusion_majority[1][0].astype(str))
st.markdown('Majority TP= ' + confusion_majority[1][1].astype(str))
st.markdown('**')

st.markdown('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.markdown('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.markdown('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.markdown('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


# In[156]:


st.subheader("KNN Classifier")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Calculate the overall accuracy on test set 

st.markdown("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))

# Calculate AUC
prob_knn = knn.predict_proba(X_test)
prob_knn = prob_knn[:, 1]

auc_knn = roc_auc_score(y_test, prob_knn)

st.markdown("AUC for KNN: {:.2f}".format(auc_knn))

y_pred = knn.predict(X_test)
confusion_majority=confusion_matrix(y_test, y_pred)
st.markdown('Majority classifier Confusion Matrix\n')
confusion_majority

st.markdown('**')
st.markdown('Majority TN= ' + confusion_majority[0][0].astype(str))
st.markdown('Majority FP= ' + confusion_majority[0][1].astype(str))
st.markdown('Majority FN= ' + confusion_majority[1][0].astype(str))
st.markdown('Majority TP= ' + confusion_majority[1][1].astype(str))
st.markdown('**')

st.markdown('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.markdown('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.markdown('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.markdown('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


# In[157]:


st.subheader("Comparison of the 3 classifiers")
st.markdown('After running all 3 models, we found out that Random Forest Classifier has the highest accuracy, AUC. ')
fpr_nb, tpr_nb, thresholds_NB = roc_curve(y_test, prob_nb) 
fpr_rf, tpr_rf, thresholds_NB = roc_curve(y_test, prob_rf) 
fpr_knn, tpr_knn, thresholds_NB = roc_curve(y_test, prob_knn) 

plt.plot(fpr_nb, tpr_nb, color='orange', label='NB')
plt.plot(fpr_rf, tpr_rf, color='blue', label='RF')
plt.plot(fpr_knn, tpr_knn, color='red', label='KNN')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()

# In[158]:


st.header("Regression")
st.markdown("Here, we use Ridge Regression, Random Forest Regressor, and Polynomial Regression to compare which model is the best to use.")
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import math 


y = df_FS['Washer_No']
X = df_FS.drop('Washer_No', axis='columns')
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.1, random_state=14)

st.subheader("Ridge Regression")
model = Ridge()
model.fit(X_train, y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predictR = model.predict(X_test)

st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictR))))


# In[159]:


st.subheader("Random Forest Regressor")
lg = RandomForestRegressor()

lg.fit(X_train, y_train)
train_score=lg.score(X_train,y_train)
test_score=lg.score(X_test,y_test)
y_predictL = lg.predict(X_test)

st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictL))))


# In[160]:


st.subheader("Polynomial Regression")
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
y_predictP = lin2.predict(poly.fit_transform(X_test))

st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictP))))


# In[161]:


st.subheader("Comparison of the 3 regression models")
st.markdown("After running all 3 models, Polynomial Regression has the least mean squared error.")
fig, axes = plt.subplots(figsize=(15,5), dpi=100)
plt.plot([n for n in range(len(X_test))], y_test, label='Actual')
plt.plot([n for n in range(len(X_test))], y_predictR, label='Ridge')
plt.plot([n for n in range(len(X_test))], y_predictL, label='Random Forest Regressor')
plt.plot([n for n in range(len(X_test))], y_predictP, label='Polynomial Regression')
plt.xlabel('Index')
plt.ylabel('Washer Number')
plt.grid()
plt.legend()
plt.show()
st.pyplot()

st.header("Clustering")
st.markdown("For convenience purpose, we read the laundrydata again to obtain specific data.")
from sklearn.cluster import KMeans 
df = pd.read_csv("LaundryData.csv")
df['datetime'] = pd.to_datetime(df['Time'])
df['hour'] = df['datetime'].dt.hour
df_Cl = df.copy()
df_Cl = df_Cl.drop(columns=['No','datetime'])
df_Cl = df_Cl.drop(columns=['Date', 'Time'])
df_Cl = df_Cl.dropna()

le = LabelEncoder()

for col in df_Cl.columns:
      if df_Cl[col].dtypes == object:
          df_Cl[col] = le.fit_transform(df_Cl[col])
y = df_Cl["Spectacles"]
X = df_Cl.drop("Spectacles", axis='columns')

KMean = KMeans(n_clusters=4)
KMean.fit(X)
label=KMean.predict(X)

df_Cl['label']=label
df_Cl

distortions = []
for i in range(1,11):
	km = KMeans(
		n_clusters=i, init='random',
		n_init=10, max_iter=300,
		tol=1e-04, random_state=1
	)
	km.fit(X)
	distortions.append(km.inertia_)

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
st.pyplot()

from kneed import KneeLocator
kl = KneeLocator(range(1, 11), distortions, curve="convex", direction="decreasing")
st.markdown("Elbow Criterion: " + str(kl.elbow))

df_new = df_Cl.copy()
df_new = df_new.drop("Spectacles", axis=1)
df_new["Spectacles"]=label

st.markdown("For example, the first cluster shows age range between 30 to 40 with hour of 0 to 10. Second cluster shows age range between 30 to 40 with hour of 11 to 23. Third cluster shows age range between 40 to 60 with hour of 0 to 12. The last cluster shows age range between 40 to 60 with hour of 13 to 23. ")

fig, axes = plt.subplots(1, 2, figsize=(13,6))

sns.scatterplot(x="Age_Range", y="hour", data=df_Cl, ax=axes[0], hue="Spectacles")
sns.scatterplot(x="Age_Range", y="hour", data=df_new, ax=axes[1], hue="Spectacles")
st.pyplot()