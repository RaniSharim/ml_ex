"""
This is a data cleansing script handling electionsdata.csv.
Input: electionsdata.csv from the current folder (script folder).
Output: 3 raw data csv and 3 csvs prepared for training and predicting

Authors: Lavi.Lazarovitz (065957383) & Aharon Sharim (052328523)
"""

import numpy as np
import pandas as pd
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.model_selection import train_test_split
# import pylab as P
from sklearn.utils import shuffle


# Importing the election data
alldata = pd.read_csv("electionsdata.csv", header=0)
alldata = shuffle(alldata)

# Slitting the data into 3 different parts: training, validation and test
train_idx = int(0.6 * len(alldata))
test_idx = train_idx + int(0.2 * len(alldata))

train_data = alldata[:train_idx]
test_data = alldata[train_idx: test_idx]
validation_data = alldata[test_idx:]

# Saving raw data before running cleaning process
train_data.to_csv("train_data.csv", index = False)
validation_data.to_csv("validation_data.csv", index = False)
test_data.to_csv("test_data.csv", index = False)

# dividing the columns into 4 categories that will be later used to decide how to remove outliers, fill gaps and scale
# the data
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
# ############################## OUTLIERS  ###########################################################################

# Recalculating outliers values in non categorical data that is more then 3 std from the mean
for key in non_class_dependand_numerical:
    mean = alldata[key].mean()
    std = alldata[key].std()
    max_val = mean + 3 * std
    min_val = mean - 3 * std
    for index, row in alldata[alldata[key] > max_val].iterrows():
        alldata.at[index, key] = np.nan
    for index, row in alldata[alldata[key] < min_val].iterrows():
        alldata.at[index, key] = np.nan

# Recalculating outliers values in categorical data that is more then 3 std from the mean
for key in class_dependand_numerical:
    for vote in alldata['Vote'].unique():
        vote_data = alldata[(alldata.Vote == vote)][[key]]
        mean = vote_data[key].mean()
        std = vote_data[key].std()
        max_val = mean + 3 * std
        min_val = mean - 3 * std
        for index, row in vote_data[vote_data[key] > max_val].iterrows():
            vote_data.at[index, key] = np.nan
        for index, row in vote_data[vote_data[key] < min_val].iterrows():
            vote_data.at[index, key] = np.nan

# ############################## FILLING GAPS  ########################################################################

# Filling features gaps based on dependent features
# First: Yearly_IncomeK - based on Avg_size_per_room
alldata['Yearly_IncomeK_Avg_size_per_room_ratio'] = alldata['Yearly_IncomeK'] / alldata['Avg_size_per_room']
ratio_to_fill = alldata['Yearly_IncomeK_Avg_size_per_room_ratio'].mean()
alldata['Yearly_IncomeK_filled'] = alldata['Yearly_IncomeK']
for index, row in alldata[alldata['Yearly_IncomeK_filled'].isnull()].iterrows():
    alldata.at[index, 'Yearly_IncomeK_filled'] = row['Avg_size_per_room'] * ratio_to_fill

# Second: Garden_sqr... based on avg_monthly_expense_on_pets...
alldata['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'] = alldata[
                                                                 'Garden_sqr_meter_per_person_in_residancy_area'] / \
                                                                  alldata['Avg_monthly_expense_on_pets_or_plants']
ratio_to_fill = alldata[
    'Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio'].mean()
alldata['Garden_sqr_meter_per_person_in_residancy_area_filled'] = alldata[
    'Garden_sqr_meter_per_person_in_residancy_area']
for index, row in alldata[alldata['Garden_sqr_meter_per_person_in_residancy_area_filled'].isnull()].iterrows():
    alldata.at[index, 'Garden_sqr_meter_per_person_in_residancy_area_filled'] = row[
                                                        'Avg_monthly_expense_on_pets_or_plants'] * ratio_to_fill

# Removing temporary columns and dependent columns
alldata['Yearly_IncomeK'] = alldata['Yearly_IncomeK_filled']
alldata = alldata.drop(['Yearly_IncomeK_Avg_size_per_room_ratio', 'Avg_size_per_room', 'Yearly_IncomeK_filled'], 1)

alldata['Garden_sqr_meter_per_person_in_residancy_area'] = alldata[
    'Garden_sqr_meter_per_person_in_residancy_area_filled']
alldata = alldata.drop(['Garden_sqr_meter_per_person_in_residancy_area_Avg_monthly_expense_on_pets_or_plants_ratio',
                        'Avg_monthly_expense_on_pets_or_plants',
                        'Garden_sqr_meter_per_person_in_residancy_area_filled'], 1)

# Fill gaps in non class dependant numerical attributes with the global median
for key in non_class_dependand_numerical:
    median = alldata[key].median()
    for index, row in alldata[alldata[key].isnull()].iterrows():
        alldata.at[index, key] = median

# Fill gaps in  class dependant numerical attributes using the median in the class Class (=Vote)
for key in class_dependand_numerical:
    for index, row in alldata[alldata[key].isnull()].iterrows():
        median = alldata[(alldata.Vote == row['Vote'])][key].median()
        alldata.at[index, key] = median

# Fill gaps in non class dependant non-numerical attributes with the global mode
for key in non_class_dependand_non_numerical:
    mode = alldata[key].dropna().mode()[0]
    for index, row in alldata[alldata[key].isnull()].iterrows():
        alldata.at[index, key] = mode

# Fill gaps class dependant non numerical data using the mode in the class Class (=Vote)
for key in class_dependand_non_numerical:
    for index, row in alldata[alldata[key].isnull()].iterrows():
        mode = alldata[(alldata.Vote == row['Vote'])][key].dropna().mode()[0]
        alldata.at[index, key] = mode



# ############################## MAPPING TO NUMBERS  #################################################################


alldata['Will_vote_only_large_party_int'] = alldata['Will_vote_only_large_party']\
                                                                    .map( {'Yes':1, 'No':-1, 'Maybe':0}).astype(int)
alldata = alldata.drop('Will_vote_only_large_party', 1)

alldata['Financial_agenda_matters_int'] = alldata['Financial_agenda_matters'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Financial_agenda_matters', 1)

alldata['Looking_at_poles_results_int'] = alldata['Looking_at_poles_results'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Looking_at_poles_results', 1)

alldata['Married_int'] = alldata['Married'].map( {'Yes':1, 'No':-1}).astype(int)
alldata = alldata.drop('Married', 1)

alldata['Gender_int'] = alldata['Gender'].map( {'Male':1, 'Female':-1}).astype(int)
alldata = alldata.drop('Gender', 1)

# map categorical values to numbers
for attr in (['Most_Important_Issue', 'Voting_Time', 'Age_group', 'Main_transportation', 'Occupation']):
    alldata[attr] = alldata[attr].astype("category")
    alldata[attr+'_int'] = alldata[attr].cat.rename_categories(range(alldata[attr].nunique())).astype(int)
    alldata = alldata.drop(attr, 1)

# ############################## SCALING  ###########################################################################

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


# ############################## FEATURE SELECTION ####################################################################
# Setting the chosen features as specified in teh exercise notes

alldata = alldata.rename(columns = {'Vote' : 'label'})


alldata = alldata.filter(['Number_of_valued_Kneset_members',
                                'Yearly_IncomeK',
                                'Overall_happiness_score',
                                'Avg_Satisfaction_with_previous_vote',
                                'Most_Important_Issue_int',
                                'Will_vote_only_large_party_int',
                                'Garden_sqr_meter_per_person_in_residancy_area',
                                'Weighted_education_rank',
                                'label'], axis='columns')

# ############################## SPLITTING DATA  ######################################################################

train_data = alldata[:train_idx]
test_data = alldata[train_idx: test_idx]
validation_data = alldata[test_idx:]

# Mapping categorical data to 1-hot columns
train_data = pd.get_dummies(train_data, columns=["Most_Important_Issue_int"], prefix=["Issue"])
validation_data = pd.get_dummies(validation_data, columns=["Most_Important_Issue_int"], prefix=["Issue"])
test_data = pd.get_dummies(test_data, columns=["Most_Important_Issue_int"], prefix=["Issue"])

# ############################## OUTPUT  ###########################################################################

train_data.to_csv("train_data_clean.csv", index = False)
validation_data.to_csv("validation_data_clean.csv", index = False)
test_data.to_csv("test_data_clean.csv", index = False)
