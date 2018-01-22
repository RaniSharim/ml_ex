#/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
import pylab as P
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from collections import Counter

train_data = pd.read_csv("train_data_clean.csv", header=0)
validation_data = pd.read_csv("validation_data_clean.csv", header=0)
test_data = pd.read_csv("test_data_clean.csv", header=0)
pred_data = pd.read_csv("ElectionsData_Pred_Features.csv", header=0)

train_val_list = [train_data,validation_data]
train_val_data = pd.concat(train_val_list)
features = train_val_data.drop(['label'], axis=1).values
target = train_val_data.label.values



clf = OneVsOneClassifier(LinearSVC(C=1.0,random_state=0))
pred = cross_val_predict(clf, features, target, cv=30,n_jobs=-1)
print(classification_report(target, pred, target_names=train_val_data.label.unique()))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    min_samples_split=5, random_state=0,n_estimators=100,n_jobs=-1,verbose=1,class_weight="balanced")
clf.fit(features,target)

val_features = validation_data.drop(['label'], axis=1).values
val_target = validation_data.label.values
predicted = clf.predict(val_features)
print(classification_report(val_target, predicted, target_names=train_val_data.label.unique()))

test_features = test_data.drop(['label'], axis=1).values
test_target = test_data.label.values
predicted = clf.predict(test_features)
print(classification_report(test_target, predicted, target_names=train_val_data.label.unique()))



bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(min_samples_split=5),
    n_estimators=600,
    learning_rate=1)

features = train_val_data.drop(['label'], axis=1).values
target = train_val_data.label.values

bdt_real.fit(features,target)
predicted = bdt_real.predict(test_features)
print(classification_report(test_target, predicted, target_names=train_val_data.label.unique()))

pred_data = pd.read_csv("//Users//admin1//Documents//MSC//ml_ex//EX_5//ElectionsData_Pred_Features.csv", header=0)
pred_data_renamed = pred_data.rename(index=str, columns={"X.Of_Household_Income": "%Of_Household_Income",
                                     "X.Time_invested_in_work": "%Time_invested_in_work",
                                     "X._satisfaction_financial_policy": "%_satisfaction_financial_policy",
                                     "Financial_balance_score_.0.1.": "Financial_balance_score_(0-1)"})

pred_data_renamed = pred_data_renamed.drop(columns="IdentityCard_Num")


############################################  prepering new data ###############################################

#alldata = pd.read_csv("//Users//admin1//Documents//MSC//ml_ex//EX_5//electionsdata.csv", header=0)

class_dependand_non_numerical = ['Will_vote_only_large_party',
                                 'Married',
                                 'Looking_at_poles_results',
                                 'Last_school_grades',
                                 'Number_of_valued_Kneset_members',
                                 'Most_Important_Issue']

non_class_dependand_non_numerical = ['Main_transportation',
                                     'Financial_agenda_matters',
                                     'Number_of_differnt_parties_voted_for',
                                     'Age_group',
                                     'Num_of_kids_born_last_10_years',
                                     'Gender',
                                     'Occupation_Satisfaction',
                                     'Occupation',
                                     'Voting_Time']

class_dependand_numerical = ['AVG_lottary_expanses',
                             'Avg_monthly_expense_when_under_age_21',
                             'Avg_Satisfaction_with_previous_vote',
                             'Garden_sqr_meter_per_person_in_residancy_area',
                             'Yearly_IncomeK',
                             # 'Avg_monthly_expense_on_pets_or_plants',
                             'Avg_monthly_household_cost',
                             'Phone_minutes_10_years',
                             # 'Avg_size_per_room',
                             'Weighted_education_rank',
                             'Avg_monthly_income_all_years',
                             'Political_interest_Total_Score',
                             'Overall_happiness_score']

non_class_dependand_numerical = ['Financial_balance_score_(0-1)',
                                 '%Of_Household_Income',
                                 'Avg_government_satisfaction',
                                 'Avg_education_importance',
                                 'Avg_environmental_importance',
                                 'Avg_Residancy_Altitude',
                                 'Yearly_ExpensesK',
                                 '%Time_invested_in_work',
                                 '%_satisfaction_financial_policy']

# Drop outliers values in non categorical data that is more then 3 std from the mean
for key in non_class_dependand_numerical:
    mean = pred_data_renamed[key].mean()
    std = pred_data_renamed[key].std()
    max_val = mean + 3 * std
    min_val = mean - 3 * std
    for index, row in pred_data_renamed[pred_data_renamed[key] > max_val].iterrows():
        pred_data_renamed.at[index, key] = np.nan
    for index, row in pred_data_renamed[pred_data_renamed[key] < min_val].iterrows():
        pred_data_renamed.at[index, key] = np.nan

for key in class_dependand_numerical:
    mean = pred_data_renamed[key].mean()
    std = pred_data_renamed[key].std()
    max_val = mean + 3 * std
    min_val = mean - 3 * std
    for index, row in pred_data_renamed[pred_data_renamed[key] > max_val].iterrows():
        pred_data_renamed.at[index, key] = np.nan
    for index, row in pred_data_renamed[pred_data_renamed[key] < min_val].iterrows():
        pred_data_renamed.at[index, key] = np.nan

print "Filling gaps"

pred_data_renamed['Yearly_IncomeK_Avg_size_per_room_ratio'] = \
    pred_data_renamed['Yearly_IncomeK'] / pred_data_renamed['Avg_size_per_room']

ratio_to_fill = pred_data_renamed['Yearly_IncomeK_Avg_size_per_room_ratio'].mean()
pred_data_renamed['Yearly_IncomeK_filled'] = pred_data_renamed['Yearly_IncomeK']
for index, row in pred_data_renamed[pred_data_renamed['Yearly_IncomeK_filled'].isnull()].iterrows():
    pred_data_renamed.at[index, 'Yearly_IncomeK_filled'] = row['Avg_size_per_room'] * ratio_to_fill

pred_data_renamed['Yearly_IncomeK'] = pred_data_renamed['Yearly_IncomeK_filled']

pred_data_renamed = pred_data_renamed.drop([
    'Yearly_IncomeK_Avg_size_per_room_ratio', 'Avg_size_per_room', 'Yearly_IncomeK_filled'], 1)

pred_data_renamed['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'] = \
    pred_data_renamed['Garden_sqr_meter_per_person_in_residancy_area'] / pred_data_renamed[
        'Avg_monthly_expense_on_pets_or_plants']

ratio_to_fill = pred_data_renamed[
    'Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'].mean()
pred_data_renamed['Garden_sqr_meter_per_person_in_residancy_area_filled'] = pred_data_renamed[
    'Garden_sqr_meter_per_person_in_residancy_area']
for index, row in pred_data_renamed[
    pred_data_renamed['Garden_sqr_meter_per_person_in_residancy_area_filled'].isnull()].iterrows():
    pred_data_renamed.at[index, 'Garden_sqr_meter_per_person_in_residancy_area_filled'] = row[
                                                                                              'Avg_monthly_expense_on_pets_or_plants'] * ratio_to_fill

pred_data_renamed['Garden_sqr_meter_per_person_in_residancy_area'] = pred_data_renamed[
    'Garden_sqr_meter_per_person_in_residancy_area_filled']
pred_data_renamed = pred_data_renamed.drop(
    ['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio',
     'Avg_monthly_expense_on_pets_or_plants', 'Garden_sqr_meter_per_person_in_residancy_area_filled'], 1)

# Fill gaps in non class dependand numerical attributes with the global median
for key in non_class_dependand_numerical:
    median = pred_data_renamed[key].median()
    for index, row in pred_data_renamed[pred_data_renamed[key].isnull()].iterrows():
        pred_data_renamed.at[index, key] = median

# Fill gaps in  class dependand numerical attributes   using the median in the class Class (=Vote)
for key in class_dependand_numerical:
    median = pred_data_renamed[key].median()
    for index, row in pred_data_renamed[pred_data_renamed[key].isnull()].iterrows():
        pred_data_renamed.at[index, key] = median

# Fill gaps in non class dependand non-numerical attributes with the global mode
for key in non_class_dependand_non_numerical:
    mode = pred_data_renamed[key].dropna().mode()[0]
    for index, row in pred_data_renamed[pred_data_renamed[key].isnull()].iterrows():
        pred_data_renamed.at[index, key] = mode

# Fill gaps class dependant non numerical data using the mode in the class Class (=Vote)
for key in class_dependand_non_numerical:
    mode = pred_data_renamed[key].dropna().mode()[0]
    for index, row in pred_data_renamed[pred_data_renamed[key].isnull()].iterrows():
        pred_data_renamed.at[index, key] = mode

print "Converting categorical values"

# convert yes/no/maybe to int values
pred_data_renamed['Will_vote_only_large_party_int'] = pred_data_renamed['Will_vote_only_large_party'].map(
    {'Yes': 1, 'No': -1, 'Maybe': 0}).astype(int)
pred_data_renamed = pred_data_renamed.drop('Will_vote_only_large_party', 1)

pred_data_renamed['Financial_agenda_matters_int'] = pred_data_renamed['Financial_agenda_matters'].map(
    {'Yes': 1, 'No': -1}).astype(int)
pred_data_renamed = pred_data_renamed.drop('Financial_agenda_matters', 1)

pred_data_renamed['Looking_at_poles_results_int'] = pred_data_renamed['Looking_at_poles_results'].map(
    {'Yes': 1, 'No': -1}).astype(int)
pred_data_renamed = pred_data_renamed.drop('Looking_at_poles_results', 1)

pred_data_renamed['Married_int'] = pred_data_renamed['Married'].map({'Yes': 1, 'No': -1}).astype(int)
pred_data_renamed = pred_data_renamed.drop('Married', 1)

pred_data_renamed['Gender_int'] = pred_data_renamed['Gender'].map({'Male': 1, 'Female': -1}).astype(int)
pred_data_renamed = pred_data_renamed.drop('Gender', 1)

# map categorical values to numbers
for attr in (['Most_Important_Issue', 'Voting_Time', 'Age_group', 'Main_transportation', 'Occupation']):
    pred_data_renamed[attr] = pred_data_renamed[attr].astype("category")
    pred_data_renamed[attr + '_int'] = pred_data_renamed[attr].cat.rename_categories(
        range(pred_data_renamed[attr].nunique())).astype(int)
    pred_data_renamed = pred_data_renamed.drop(attr, 1)

# Scale data to 0..1 range
# These attributes are 1> >10, and more or less uniform, so just scale them back by a factor of 10 (Decimal Scaling)
pred_data_renamed['Occupation_Satisfaction'] = pred_data_renamed['Occupation_Satisfaction'].map(lambda x: x / 10)
pred_data_renamed['Avg_government_satisfaction'] = pred_data_renamed['Avg_government_satisfaction'].map(
    lambda x: x / 10)
pred_data_renamed['Avg_education_importance'] = pred_data_renamed['Avg_education_importance'].map(lambda x: x / 10)
pred_data_renamed['Avg_environmental_importance'] = pred_data_renamed['Avg_environmental_importance'].map(
    lambda x: x / 10)
pred_data_renamed['Avg_Residancy_Altitude'] = pred_data_renamed['Avg_Residancy_Altitude'].map(lambda x: x / 10)

# This is just percent in 0..100, so we can just divide by 100 (Decimal Scaling)
pred_data_renamed['%Time_invested_in_work'] = pred_data_renamed['%Time_invested_in_work'].map(lambda x: x / 100)
pred_data_renamed['%_satisfaction_financial_policy'] = pred_data_renamed['%_satisfaction_financial_policy'].map(
    lambda x: x / 100)
pred_data_renamed['Last_school_grades'] = pred_data_renamed['Last_school_grades'].map(lambda x: x / 100)

# Yearly_ExpensesK is nearly uniform, so we use Min-Max
minv = pred_data_renamed['Yearly_ExpensesK'].min()
maxv = pred_data_renamed['Yearly_ExpensesK'].max()
pred_data_renamed['Yearly_ExpensesK'] = pred_data_renamed['Yearly_ExpensesK'].map(
    lambda v: (v - minv) / (maxv - minv) * 2 - 1)

# All the rest of the features, map using z-score
zscore = ['Garden_sqr_meter_per_person_in_residancy_area',
          'Number_of_valued_Kneset_members',
          'AVG_lottary_expanses',
          'Avg_Satisfaction_with_previous_vote',
          'Yearly_IncomeK',
          'Avg_monthly_expense_when_under_age_21',
          'Avg_monthly_household_cost',
          'Phone_minutes_10_years',
          'Weighted_education_rank',
          'Avg_monthly_income_all_years',
          'Political_interest_Total_Score',
          'Number_of_differnt_parties_voted_for',
          'Overall_happiness_score',
          'Num_of_kids_born_last_10_years']

for attr in zscore:
    attr_std = pred_data_renamed[attr].std()
    attr_mean = pred_data_renamed[attr].mean()
    pred_data_renamed[attr] = pred_data_renamed[attr].map(lambda v: (v - attr_mean) / attr_std)

pred_data_renamed = pred_data_renamed.filter(['Number_of_valued_Kneset_members',
                                              'Yearly_IncomeK',
                                              'Overall_happiness_score',
                                              'Avg_Satisfaction_with_previous_vote',
                                              'Most_Important_Issue_int',
                                              'Will_vote_only_large_party_int',
                                              'Garden_sqr_meter_per_person_in_residancy_area',
                                              'Weighted_education_rank',
                                              'label'], axis='columns')



fig, axes = pyplot.subplots(6, 2)
x = 0
for feature in pred_data_renamed.dtypes[pred_data_renamed.dtypes=='float64'].keys():
    pred_data_renamed[feature].hist( bins=20, ax=axes[x,0])
    pred_data[feature].hist(bins=20, ax=axes[x,1])
    x += 1
pyplot.show()

pred_data_final = pd.get_dummies(pred_data_renamed, columns=["Most_Important_Issue_int"], prefix=["Issue"])

predicted_votes = clf.predict(pred_data_final.values)


vote_count = Counter(predicted_votes)

print vote_count

for key in vote_count:
    print "%s: %f" %(key,float(vote_count[key] / 100.0))

id_data = pred_data[['IdentityCard_Num']]
pred_vote_df = pd.DataFrame(data=predicted_votes, columns=["PredictVote"])
id_vote = pd.concat([id_data, pred_vote_df], axis=1)

id_vote.to_csv("Vote_prediction.csv", index = False)
pred_data_final.to_csv("clean_features.csv", index = False)