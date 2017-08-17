#!/usr/bin/python 

import pandas as pd
import numpy as np
import json
from patsy.contrasts import Treatment
from patsy.contrasts import Diff
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Set ipython's max column display
pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 5000)

data =[]
for line in open('projectStatsAll.json') :
    data.append(json.loads(line))

df = pd.DataFrame(data)
lists = []

for index,row in df.iterrows() :
    for dic in row['orgs'] :
        dic['projectId'] = row['projectId']
        dic['projectDuration'] = row['projectDuration']
        dic['mailCount'] = row['mailCount']
        dic['projectStart_ts'] = row['start_ts']
        dic['projectEnd_ts'] = row['end_ts']
        lists.append(dic)

#print lists
orgs = pd.DataFrame(lists)

orgs['orgStart_ts'] = pd.to_datetime(orgs['orgStart_ts'], unit='ms')
orgs['orgEnd_ts'] = pd.to_datetime(orgs['orgEnd_ts'], unit='ms')
orgs['orgDuration'] = orgs['orgDuration'] / 1000
orgs['projectStart_ts'] = pd.to_datetime(orgs['projectStart_ts'], unit='ms')
orgs['projectEnd_ts'] = pd.to_datetime(orgs['projectEnd_ts'], unit='ms')
orgs['projectDuration'] = orgs['projectDuration'] / 1000 # scale duration times in microseconds to seconds
projDuration_mean = orgs['projectDuration'].mean()
orgs['orgDurationNorm'] = orgs['orgDuration'] / projDuration_mean
mailCount_mean = orgs['mailCount'].mean()
#print mailCount_mean 
orgs['orgMailCountNorm'] = orgs['orgMailCount'] / mailCount_mean
#print orgs.head(10)

# correlations between project duration and org mail count/

# list site_instructions/variations counts  by Org Id


# top n organizations by Org Mail Count
n = 20

topOrgIdsNorm = orgs.sort_values('orgMailCountNorm', ascending=False).head(n) #  groupby('orgId').orgMailCount.sum()
print topOrgIdsNorm

#print topOrgIds[['orgId','orgMailCountNorm']]
topOrgIds = pd.DataFrame(topOrgIdsNorm['orgId'], columns=['orgId']).reset_index()
print topOrgIds

topOrgs = orgs[orgs['orgId'].isin(topOrgIds['orgId'].values)]
print topOrgs.sort_values('orgId', ascending=False)
topOrgIds['orgIdIndex'] = topOrgIds.index
del topOrgIds['index']
print topOrgIds  
print topOrgs.head(10)

topProjIds = topOrgs.groupby('projectId').projectId.count().sort_values(ascending=False).head(n)
print topProjIds

print topOrgs.head(10)
print 'top' + str(n) + ' Organizations : '
print topOrgIds['orgId'].values

# split dataframe into training and test sets

df = pd.merge(topOrgs[['projectId', 'projectDuration', 'orgId']], topOrgIds, how='left', on=['orgId'])
#print df.head(10)

#topProjIds = df.groupby('projectId').count().sort_values(ascending=False).head(10)
#print topProjIds

train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
test = test.drop_duplicates(subset = 'orgId')  #[test['projectId'].isin([topProjIds.index[1]])]

print train.head(20)

orgIdPDMean = pd.DataFrame(df.groupby('orgId', as_index=False)['projectDuration'].mean())
orgIdPDMean.rename(columns={'projectDuration':'projectDurationMean'}, inplace=True)
orgIdPDMean['projectDurationMean'] =  orgIdPDMean['projectDurationMean'] / (60 * 60 *24)
#print orgIdPDMean

# joining on project duration mean 
test = pd.merge(test, orgIdPDMean, how='inner', on=['orgId'])
test['index'] = test.index
#print '\n test set:'
#print test.head(10)

print train
levels = [x+1 for x in range(n)]
print '\n levels : ' + str(levels)
#contrast = Treatment(reference=0).code_without_intercept(levels)
contrast = Diff().code_without_intercept(levels)
print(contrast.matrix)
#print(contrast.matrix)
#print(contrast.matrix[train['orgIdIndex']-1, :][:20])
print sm.categorical(train['orgIdIndex'].values)

mod = ols('projectDuration ~ C(orgIdIndex, Treatment)', data=train)
res = mod.fit()
#print(res.summary())

predictions = pd.DataFrame(res.predict(test), columns=['predProjectDuration'])

predictions['index'] = predictions.index
#print predictions
predictions = pd.merge(predictions, test, how='inner', on=['index'])
#predictions['predProjectDuration'] = predictions['predProjectDuration'].astype('timedelta64[s]')
predictions['plotIndex'] = predictions.index

print '\n predictions:'
print predictions

#plot project durations for top Org Ids
#print '\n Top Org Id: ' + str(topOrgIds['orgId'].tolist()[:])

X = (predictions['orgId'][predictions['orgId'].isin(topOrgIds['orgId'].values)].sort_values(ascending=True).head(20)).tolist()

print 'X: '
print X
X_ticks = range(len(X))
X_labels = X
y_pred = (predictions['predProjectDuration'][predictions['orgId'].isin(topOrgIds['orgId'].values)] / (60 * 60 *24)).head(20).tolist()
print 'y_pred: '
print y_pred
y_act = (predictions['projectDuration'][predictions['orgId'].isin(topOrgIds['orgId'].values)] / (60 * 60 *24)).head(20).tolist()
y_mean = (predictions['projectDurationMean'][predictions['orgId'].isin(topOrgIds['orgId'].values)]).head(20).tolist()
print 'y_mean: '
print y_mean

se = []
se_mean = []

for i in range(len(y_pred)) :
    
    print type(y_pred[i])
    print type(y_act[i])
    se.append((y_pred[i] - y_act[i])**2.)
    se_mean.append((y_mean[i] - y_act[i])**2.)
rmse = np.sqrt(np.mean(se[:]))
print 'rmse : ' + str(rmse)
rmse_mean = np.sqrt(np.mean(se_mean[:]))
print 'rmse_mean: ' + str(rmse_mean)
fig, ax = plt.subplots()

plt.scatter(X_ticks, y_pred, color='b')
plt.scatter(X_ticks, y_act, color='g')
plt.scatter(X_ticks, y_mean, color='r')

plt.xlabel('Organization ID')
plt.ylabel('Duration (days)')
ax.set_xticks(X_ticks)
ax.set_xticklabels(X_labels, rotation=90)
fig.savefig('./figures/projectDurationByOrg.png')
#
plt.show()
   