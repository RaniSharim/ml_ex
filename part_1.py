import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

print "Loading data"

#alldata = pd.read_csv("ElectionsData.csv", header=0, index_col=0)
alldata = pd.read_csv("ElectionsData.csv", header=0)

#Fill gaps in Yearly_IncomeK using the mean ratio and the values in Avg_size_per_room
print "Filling gaps"

alldata['Yearly_IncomeK_Avg_size_per_room_ratio'] = alldata['Yearly_IncomeK'] / alldata['Avg_size_per_room']
ratio_to_fill = alldata['Yearly_IncomeK_Avg_size_per_room_ratio'].mean()
alldata['Yearly_IncomeK_filled'] = alldata['Yearly_IncomeK']
for index, row in alldata[alldata['Yearly_IncomeK_filled'].isnull()].iterrows():
    alldata.at[index, 'Yearly_IncomeK_filled'] = row['Avg_size_per_room'] * ratio_to_fill

alldata['Yearly_IncomeK'] = alldata['Yearly_IncomeK_filled']
alldata = alldata.drop(['Yearly_IncomeK_Avg_size_per_room_ratio', 'Avg_size_per_room', 'Yearly_IncomeK_filled'], 1)

alldata['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'] = alldata['Garden_sqr_meter_per_person_in_residancy_area'] / alldata['Avg_monthly_expense_on_pets_or_plants']
ratio_to_fill = alldata['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'].mean()
alldata['Garden_sqr_meter_per_person_in_residancy_area_filled'] = alldata['Garden_sqr_meter_per_person_in_residancy_area']
for index, row in alldata[alldata['Garden_sqr_meter_per_person_in_residancy_area_filled'].isnull()].iterrows():
    alldata.at[index, 'Garden_sqr_meter_per_person_in_residancy_area_filled'] = row['Avg_monthly_expense_on_pets_or_plants'] * ratio_to_fill

alldata['Garden_sqr_meter_per_person_in_residancy_area'] = alldata['Garden_sqr_meter_per_person_in_residancy_area_filled']
alldata = alldata.drop(['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio','Avg_monthly_expense_on_pets_or_plants','Garden_sqr_meter_per_person_in_residancy_area_filled'], 1)

# This the the breakdown of the attributes, to numberical vs categorical,
# and those which seems class dependant distribution to those which aren't
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
                             #'Avg_monthly_expense_on_pets_or_plants',
                             'Avg_monthly_household_cost',
                             'Phone_minutes_10_years',
                             #'Avg_size_per_room',
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

#Fill gaps in non class dependand numerical attributes with the global median
for key in non_class_dependand_numerical:
    median = alldata[key].median()
    for index, row in alldata[alldata[key].isnull()].iterrows():
        alldata.at[index, key] = median

#Fill gaps in  class dependand numerical attributes   using the median in the class Class (=Vote)
for key in class_dependand_numerical:
    for index, row in alldata[alldata[key].isnull()].iterrows():
        median = alldata[(alldata.Vote == row['Vote'])][key].median()
        alldata.at[index, key] = median

#Fill gaps in non class dependand non-numerical attributes with the global mode
for key in non_class_dependand_non_numerical:
    mode = alldata[key].dropna().mode()[0]
    for index, row in alldata[alldata[key].isnull()].iterrows():
        alldata.at[index, key] = mode

#Fill gaps class dependant non numerical data using the mode in the class Class (=Vote)
for key in class_dependand_non_numerical:
    for index, row in alldata[alldata[key].isnull()].iterrows():
        mode = alldata[(alldata.Vote == row['Vote'])][key].dropna().mode()[0]
        alldata.at[index, key] = mode

#print alldata.isnull().values.any()

print "Converting categorical values"

#convert yes/no/maybe to int values
alldata['Will_vote_only_large_party_int'] = alldata['Will_vote_only_large_party'].map( {'Yes':1, 'No':-1, 'Maybe':0}).astype(int)
alldata = alldata.drop('Will_vote_only_large_party', 1)

alldata['Financial_agenda_matters_int'] = alldata['Financial_agenda_matters'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Financial_agenda_matters', 1)

alldata['Looking_at_poles_results_int'] = alldata['Looking_at_poles_results'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Looking_at_poles_results', 1)

alldata['Married_int'] = alldata['Married'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Married', 1)

alldata['Gender_int'] = alldata['Gender'].map( {'Male':1, 'Female':-1}).astype(int)
alldata = alldata.drop('Gender', 1)

#print alldata.isnull().values.any()

# Split categorical attributes to categorical attributes, 1/0 per category
for attr in (['Most_Important_Issue', 'Voting_Time', 'Age_group', 'Main_transportation', 'Occupation']):
    alldata[attr] = alldata[attr].astype("category")
    alldata[attr+'_int'] = alldata[attr].cat.rename_categories(range(alldata[attr].nunique())).astype(int)
    for i in range(alldata[attr+'_int'].nunique()):
        alldata[attr+"_"+str(i)] = alldata[attr+'_int'].map(lambda x: 1 if x == i else 0).astype(int)
    alldata = alldata.drop(attr, 1)
    alldata = alldata.drop(attr+'_int', 1)

#print alldata.dtypes
print "Removing outliers"

# Drop ouliers using k-nearst neightbores, assume we have 0.1% outliers (as is the default)
outlier_factor = 0.01
number_of_outliers = len(alldata) * outlier_factor
outlier_classifier = LocalOutlierFactor(n_neighbors=20, contamination=outlier_factor)
prediction = outlier_classifier.fit_predict(alldata.drop('Vote',1), alldata.Vote.values)
mask = pd.DataFrame(prediction)[0].map(lambda x: x==-1)
alldata = alldata.mask(mask).dropna()
print "new len: " + str(len(alldata))

print "Scaling data"

# Scale data to 0..1 range
#These attributes are 1> >10, and more or less uniform, so just scale them back by a factor of 10 (Decimal Scaling)
alldata['Occupation_Satisfaction'] = alldata['Occupation_Satisfaction'].map(lambda x: x/10)
alldata['Avg_government_satisfaction'] = alldata['Avg_government_satisfaction'].map(lambda x: x/10)
alldata['Avg_education_importance'] = alldata['Avg_education_importance'].map(lambda x: x/10)
alldata['Avg_environmental_importance'] = alldata['Avg_environmental_importance'].map(lambda x: x/10)
alldata['Avg_Residancy_Altitude'] = alldata['Avg_Residancy_Altitude'].map(lambda x: x/10)

#This is just percent in 0..100, so we can just divide by 100 (Decimal Scaling)
alldata['%Time_invested_in_work'] = alldata['%Time_invested_in_work'].map(lambda x: x/100)
alldata['%_satisfaction_financial_policy'] = alldata['%_satisfaction_financial_policy'].map(lambda x: x/100)
alldata['Last_school_grades'] = alldata['Last_school_grades'].map(lambda x: x/100)

#Yearly_ExpensesK is nearly uniform, so we use Min-Max
minv = alldata['Yearly_ExpensesK'].min()
maxv = alldata['Yearly_ExpensesK'].max()
alldata['Yearly_ExpensesK'] = alldata['Yearly_ExpensesK'].map(lambda v: (v - minv)/(maxv - minv)*2 - 1)

#All the rest of the features, map using z-score
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
    attr_std = alldata[attr].std()
    attr_mean = alldata[attr].mean()
    alldata[attr] = alldata[attr].map(lambda v: (v - attr_mean)/attr_std)

from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif

data_without_votes = alldata.drop(['Vote'], axis=1)
data_X = data_without_votes.values
data_Y = alldata.Vote.values

print "Find features by variance and mutual information"

selector = SelectPercentile(f_classif, percentile=60)
selector.fit(data_X, data_Y)
support = selector.get_support()
f_classif_selected = []
f_classif_full = []

for idx, col in enumerate(data_without_votes.columns):
    if support[idx]:
        f_classif_selected.append(col)
    f_classif_full.append((col,selector.scores_[idx]))

# print "column scores:"
# print f_classif_full
# print "selected:"
# print f_classif_selected

selector = SelectPercentile(mutual_info_classif, percentile=60)
selector.fit(data_X, data_Y)
support = selector.get_support()

mutual_info_classif_selected = []
mutual_info_classif_full = []

for idx, col in enumerate(data_without_votes.columns):
    if support[idx]:
        mutual_info_classif_selected.append(col)
    mutual_info_classif_full.append((col,selector.scores_[idx]))

# print "column scores:"
# print mutual_info_classif_full
# print "selected:"
# print mutual_info_classif_selected


print "Find features by Backward elimination"
#split to train, test, validation
from sklearn.model_selection import train_test_split
#probably want to remove this to train a full model
train, test = train_test_split(alldata, test_size=0.9)
data_X = train.drop(['Vote'], axis=1).values
data_Y = train.Vote.values

# This takes a huge amount of time on the full data
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
svc = SVC(kernel="linear", C=1)
rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
rfecv.fit(data_X, data_Y)
support = rfecv.get_support()
# print support
rfecv_selected = []
rfecv_full = []

for idx, col in enumerate(data_without_votes.columns):
    if support[idx]:
        rfecv_selected.append(col)
    rfecv_full.append((col,selector.scores_[idx]))

# print "column scores:"
# print rfecv_full
# print "selected:"
# print rfecv_selected


#These are the features we want that we know that has potentiall some data
selected_features = ['Will_vote_only_large_party_int',
                     'Married_int',
                     'Looking_at_poles_results_int',
                     'Last_school_grades',
                     'Number_of_valued_Kneset_members',
                     'AVG_lottary_expanses',
                     'Avg_monthly_expense_when_under_age_21',
                     'Avg_Satisfaction_with_previous_vote',
                     'Garden_sqr_meter_per_person_in_residancy_area',
                     'Yearly_IncomeK',
                     'Avg_monthly_expense_on_pets_or_plants',
                     'Avg_monthly_household_cost',
                     'Phone_minutes_10_years',
                     'Avg_size_per_room',
                     'Weighted_education_rank',
                     'Avg_monthly_income_all_years',
                     'Political_interest_Total_Score',
                     'Overall_happiness_score']

for feature in rfecv_selected:
    if ((feature in f_classif_selected or feature in mutual_info_classif_selected) and (feature not in selected_features)):
        selected_features.append(feature)

#split to train, test, validation
#from sklearn.model_selection import train_test_split
#20% to test 
train, test = train_test_split(alldata, test_size=0.2)
# ~20% validation
train, validation = train_test_split(train, test_size=0.25)

#Write everything to file
train.to_csv("train.csv")
test.to_csv("test.csv")
validation.to_csv("validation.csv")
pd.DataFrame(selected_features).to_csv("feature.csv", index = False, header = False)