# Senin 15 Juli 2019

# Soal 1 - Diagnosis Kesuburan

import pandas as pd

df = pd.read_csv('fertility.csv')
df = df.rename(columns=
    {'Childish diseases': 'childish_diseases',
    'Accident or serious trauma': 'accident_or_serious_trauma', 
    'Surgical intervention': 'surgical_intervention',
    'High fevers in the last year': 'high_fevers_in_the_last_year',
    'Frequency of alcohol consumption': 'frequency_of_alcohol_consumption',
    'Smoking habit': 'smoking_habit',
    'Number of hours spent sitting per day': 'number_of_hours_spent_sitting_per_day'})


# ============================================================== LABELLING =======================================================================================================================
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['childish_diseases'] = label.fit_transform(df['childish_diseases'])
# print(label.classes_)               # ['no' 'yes']

df['accident_or_serious_trauma'] = label.fit_transform(df['accident_or_serious_trauma'])
# print(label.classes_)               # ['no' 'yes']

df['surgical_intervention'] = label.fit_transform(df['surgical_intervention'])
# print(label.classes_)               # ['no' 'yes']

df['high_fevers_in_the_last_year'] = label.fit_transform(df['high_fevers_in_the_last_year'])
# print(label.classes_)               # ['less than 3 months ago' 'more than 3 months ago' 'no']

df['frequency_of_alcohol_consumption'] = label.fit_transform(df['frequency_of_alcohol_consumption'])
# print(label.classes_)               # ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']

df['smoking_habit'] = label.fit_transform(df['smoking_habit'])
# print(label.classes_)               # ['daily' 'never' 'occasional']

#=======================================split feature X and Target Y

dfx = df.drop(['Season', 'Diagnosis'], axis=1)
# print(dfx.iloc[0])

dfy = df['Diagnosis']

# ============================================================== SPLITTING DATA =======================================================================================================================
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(
    dfx, dfy, test_size= 0.1
)

# print(len(xtrain))
# print(len(xtest))
# print(len(ytrain))
# print(len(ytest))

# ============================================================== MACHINE LEARNING =======================================================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

model1 = LogisticRegression(solver='liblinear', multi_class='auto')
model2 = RandomForestClassifier(n_estimators=50)
model3 = ExtraTreesClassifier(n_estimators=50)

model1.fit(xtrain,ytrain)
model2.fit(xtrain,ytrain)
model3.fit(xtrain,ytrain)

#================================================================== ARIN

# prediction for Arin
# Age                                      29    'age = 29'
# childish_diseases                         0    'no child diseases'
# accident_or_serious_trauma                0    'no accident / serious trauma
# surgical_intervention                     0    'no surgical intevention'
# high_fevers_in_the_last_year              2    'no high fevers in the last year
# frequency_of_alcohol_consumption          3    'several times a day'
# smoking_habit                             0    'daily'
# number_of_hours_spent_sitting_per_day     5    '5 hours

# logistic model
p = model1.predict([[29,0,0,0,2,3,0,5]])
print('Arin, prediksi kesuburan: ' + p[0] + ' (Logistic Regression)')

# random forest
r = model2.predict([[29,0,0,0,2,3,0,5]])
print('Arin, prediksi kesuburan: ' + r[0] + ' (Random Forest)') 

# extreme random forest
e = model2.predict([[29,0,0,0,2,3,0,5]])
print('Arin, prediksi kesuburan: ' + e[0] + ' (Extreme Random Forest)') 

print('=========================================================================')
#================================================================== BEBI

# prediction for Bebi
# Age                                      31    'age = 31'
# childish_diseases                         0    'no child diseases'
# accident_or_serious_trauma                1    'there's accident'
# surgical_intervention                     1    'there's surgical intervention (amputation)'
# high_fevers_in_the_last_year              2    'no high fevers in the last year'
# frequency_of_alcohol_consumption          4    'several times a week'
# smoking_habit                             1    'never'
# number_of_hours_spent_sitting_per_day     0    '0 hours (no information of how many hours spent sitting per day)

# logistic model
p1 = model1.predict([[31,0,1,1,2,4,1,0]])
print('Bebi, prediksi kesuburan: ' + p1[0] + ' (Logistic Regression)')

# random forest
r1 = model2.predict([[31,0,1,1,2,4,1,0]])
print('Bebi, prediksi kesuburan: ' + r1[0] + ' (Random Forest)')

# extreme random forest
e1 = model3.predict([[31,0,1,1,2,4,1,0]])
print('Bebi, prediksi kesuburan: ' + e1[0] + ' (Extreme Random Forest)')

print('=========================================================================')
#================================================================== CACA

# prediction for Caca
# Age                                      25    'age = 25'
# childish_diseases                         1    'there's child disease'
# accident_or_serious_trauma                0    'no accident'
# surgical_intervention                     0    'no surgical intervention'
# high_fevers_in_the_last_year              0    'high fevers in the last year: less than 3 months ago'
# frequency_of_alcohol_consumption          1    'hardly ever or never'
# smoking_habit                             1    'never'
# number_of_hours_spent_sitting_per_day     7    '7 hours'

# logistic model
p2 = model1.predict([[25,1,0,0,0,1,1,7]])
print('Caca, prediksi kesuburan: ' + p2[0] + ' (Logistic Regression)')

# random forest
r2 = model2.predict([[25,1,0,0,0,1,1,7]])
print('Caca, prediksi kesuburan: ' + r2[0] + ' (Random Forest)')

# extreme random forest
e2 = model3.predict([[25,1,0,0,0,1,1,7]])
print('Caca, prediksi kesuburan: ' + e2[0] + ' (Extreme Random Forest)')

print('=========================================================================')
#================================================================== DINI

# prediction for Dini
# Age                                      28    'age = 28'
# childish_diseases                         0    'no child disease'
# accident_or_serious_trauma                1    'there's accident'
# surgical_intervention                     1    'there's surgical intervention
# high_fevers_in_the_last_year              2    'no high fever'
# frequency_of_alcohol_consumption          1    'hardly ever or never'
# smoking_habit                             0    'daily'
# number_of_hours_spent_sitting_per_day     24    '24 hours'

# logistic model
p3 = model1.predict([[28,0,1,1,2,1,0,24]])
print('Dini, prediksi kesuburan: ' + p3[0] + ' (Logistic Regression)')

# random forest
r3 = model2.predict([[28,0,1,1,2,1,0,24]])
print('Dini, prediksi kesuburan: ' + r3[0] + ' (Random Forest)')

# extreme random forest
e3 = model3.predict([[28,0,1,1,2,1,0,24]])
print('Dini, prediksi kesuburan: ' + e3[0] + ' (Extreme Random Forest)')

print('=========================================================================')
#================================================================== ENNO

# prediction for Enno
# Age                                      42    'age = 42'
# childish_diseases                         1    'child disease'
# accident_or_serious_trauma                0    'no accident'
# surgical_intervention                     0    'no surgical intervention'
# high_fevers_in_the_last_year              1    'there's high fever: more than 3 months ago due to bronkitis'
# frequency_of_alcohol_consumption          1    'hardly ever or never'
# smoking_habit                             1    'never'
# number_of_hours_spent_sitting_per_day     8    '8 hours'

# logistic model
p4 = model1.predict([[42,1,0,0,1,1,1,8]])
print('Enno, prediksi kesuburan: ' + p4[0] + ' (Logistic Regression)')

# random forest
r4 = model2.predict([[42,1,0,0,1,1,1,8]])
print('Enno, prediksi kesuburan: ' + r4[0] + ' (Random Forest)')

# extreme random forest
e4 = model3.predict([[42,1,0,0,1,1,1,8]])
print('Enno, prediksi kesuburan: ' + e4[0] + ' (Extreme Random Forest)')

print('=========================================================================')
