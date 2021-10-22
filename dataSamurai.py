import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Read in the spreadsheets
calls = pd.read_csv('calls.csv')
signups = pd.read_csv('signups.csv')
leads = pd.read_csv('leads.csv')

#Determine which agent made the most calls
most_calls = calls['Agent'].value_counts().idxmax()
print("Agent",most_calls,"made the most calls")

#Determine how many calls were received on average for those who signed
#Get all leads who signed up
signed_leads = leads[leads.Name.isin(signups.Lead)]
#Get the phone numbers in calls who signed
signed_calls = calls[calls['Phone Number'].isin(signed_leads['Phone Number'])]
average_calls = signed_calls['Phone Number'].value_counts().mean()
print("Of those signed up, average number of calls was", average_calls)

#Which agent had the most signups
int_signed_calls = signed_calls[signed_calls['Call Outcome'] == 'INTERESTED']
print("Agent",int_signed_calls['Agent'].value_counts().idxmax(),"had the most signups")
'''Assumed that if the call outcome was INTERESTED,
then the agent should take credit if the number ended up being signed.
Therefore assume that only one call required for signing up after interest.'''

#Determine who had most signups per call
q_calls = calls['Agent'].value_counts()
signed = int_signed_calls['Agent'].value_counts()

bestAgent = {}
for agent in calls['Agent'].unique():
    bestAgent[agent] = signed[agent]/q_calls[agent]

print('The agent with the highest signed per calls ratio was',max(bestAgent,key=bestAgent.get))

#Determine which region most likely to be "interested"?
tmp = 0
bestRegion = ""
#Loop through regions
import sys
for r in leads['Region'].unique():

    #Get the leads table for the region
    regTable = leads[leads['Region'] == r]

    #Get the rows from calls which correspond 
    regCalls = calls[calls['Phone Number'].isin(regTable['Phone Number'])]

    #Determine number of INTERESTED calls
    interest = len(regCalls[regCalls['Call Outcome'] == 'INTERESTED'])
    notinterest = len(regCalls[regCalls['Call Outcome'] == 'NOT INTERESTED'])
    int_not = interest + notinterest
    ratio = interest/int_not

    if ratio > tmp:
        tmp = ratio
        bestRegion = r

print('The most likely region to be interested is:',bestRegion)

#Given lead signed, determine which region most likely to be approved.
tmp = 0
#Perform inner join to get leads already signed up
joined = signups.set_index('Lead').join(leads.set_index('Name'))

#For statistical significance
ratioList = []
for r in joined['Region'].unique():
    approved = len(joined[(joined['Region'] == r) & (joined['Approval Decision'] == 'APPROVED') ])
    total = len(joined[joined['Region'] == r])
    ratio = approved / total
    ratioList.append(ratio)
    if(ratio > tmp):
        tmp = ratio
        bestRegion = r

print('The most likely region to be approved given interested is:',bestRegion, 'with ratio of',tmp)

#Want to determine whether the best ratio is statistically separate from the others
#Determine std deviation and mean of the remaining data
std = np.std(ratioList)
mean = np.average(ratioList)

#Determine how many standard deviation is away from the mean
#This is the so-called Z-score if assuming a Gaussian distribution
zscore = np.abs(tmp-mean)/std

#Now see the probability of this zscore for a one tailed distribution 
print('There is a', stats.norm.sf(zscore)*100.0,'% chance this belongs to the distribution')
print('For a 10% significance region this is statistically significant')
