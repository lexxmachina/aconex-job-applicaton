#!/usr/bin/python 

import pandas as pd
import numpy as np
import json
from patsy.contrasts import Treatment
from patsy.contrasts import Diff
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

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
orgs['orgMailCountNorm'] = orgs['orgMailCount'] / mailCount_mean

# list site_instructions/variations counts by Org Id

orgSiteMailMean = orgs.groupby('orgId').orgSiteMailCount.mean().sort_values(ascending=False)
print orgSiteMailMean.head(10)
orgVarMailMean = orgs.groupby('orgId').orgVarMailCount.mean().sort_values(ascending=False)
print orgVarMailMean.head(10)

# subdivide data according to quantiles of Organisation mail durations   
orgDurationMeans = orgs.groupby('orgId').orgDuration.mean().sort_values(ascending=False)
print orgDurationMeans.head(10)

#orgDurationsQuants = pd.qcut(orgDurationMeans, [0, .25, .5, .75, 1.], duplicates='drop') 
orgDurationsQuants = []
orgDurationsQuants.append(-0.01)
orgDurationsQuants.append(orgDurationMeans.quantile(0.20))
orgDurationsQuants.append(orgDurationMeans.quantile(0.60))
orgDurationsQuants.append(orgDurationMeans.quantile(1.0))
print orgDurationsQuants

orgDurationMeans = orgDurationMeans.to_frame().reset_index()

orgDurationMeans.loc[:, 'bins'] = pd.cut(orgDurationMeans['orgDuration'], bins=orgDurationsQuants, labels=[1,2,3])
cat_columns = orgDurationMeans.select_dtypes(['category']).columns

orgDurationMeans[cat_columns] = orgDurationMeans[cat_columns].apply(lambda x: x.cat.codes)

# split dataframe into training and test sets

df = pd.merge(orgs[['projectId', 'projectDuration', 'orgId']], orgDurationMeans, how='left', on=['orgId'])

train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
test = test.drop_duplicates(subset = 'orgId')  #[test['projectId'].isin([topProjIds.index[1]])]
n_bins = train['bins'].nunique()
test['index'] = test.index

print '\n train: ' 
print train['bins'].head(10)

levels = [x+1 for x in range(n_bins)]
print '\n levels : ' + str(levels)
contrast = Treatment(reference=0).code_without_intercept(levels)
#contrast = Diff().code_without_intercept(levels)
print(contrast.matrix)
print(contrast.matrix[train['bins']-1, :][:20])
print sm.categorical(train['bins'].values)

mod = ols('projectDuration ~ C(bins, Treatment)', data=train)
res = mod.fit()
print(res.summary())

predictions = pd.DataFrame(res.predict(test), columns=['predProjectDuration'])
predictions['projectDurationMean'] = df['projectDuration'].mean()
predictions['index'] = predictions.index
predictions = pd.merge(predictions, test, how='inner', on=['index'])
predictions['plotIndex'] = predictions.index

print '\n predictions:'
print predictions.head(20)

# scatter plot 
X = predictions['orgId'].head(20).tolist()
X_bins = predictions['bins'].head(20).tolist()
X_ticks = range(len(X))
X_labels = X
y_pred = (predictions['predProjectDuration'] / (60 * 60 * 24)).head(20).tolist()
y_act = (predictions['projectDuration'] / (60 * 60 * 24)).head(20).tolist()
y_mean = (predictions['projectDurationMean'] / (60 * 60 * 24)).head(20).tolist()

se = []
se_mean = []

for i in range(len(y_pred)) :  
    se.append((y_pred[i] - y_act[i])**2.)
    se_mean.append((y_mean[i] - y_act[i])**2.)
rmse = np.sqrt(np.mean(se[:]))
print 'rmse : ' + str(rmse)
rmse_mean = np.sqrt(np.mean(se_mean[:]))
print 'rmse_mean: ' + str(rmse_mean)

# bar plot
fig, ax = plt.subplots()
plt.scatter(X_ticks, y_pred, color='b', label='predictions')
plt.scatter(X_ticks, y_act, color='g', label='actual')
plt.scatter(X_ticks, y_mean, color='r', label='mean')

plt.xlabel('Organization ID')
plt.ylabel('Duration (days)')
plt.title('Project Duration for selected organisations')
ax.set_xticks(X_ticks)
ax.set_xticklabels(X_labels, rotation=90)
plt.legend()
plt.tight_layout()
fig.savefig('./figures/projectDurationByOrgMailDuration.png')
plt.show()

n_groups = n_bins

pd_pred = sorted((predictions['predProjectDuration']/ (60 * 60 * 24)).unique().tolist())
pd_mean = (predictions['projectDurationMean']/ (60 * 60 * 24)).unique().tolist() * n_bins

# create bar plot
fig, ax = plt.subplots()
index = np.arange(n_bins)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, pd_pred, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Predicted Project Duration')

plt.axhline(pd_mean[0], color='k', linestyle='solid')
plt.xlabel('Mail Duration Quantiles')
plt.ylabel('Project Duration (days)')
plt.title('Predicted Project Duration by Mail Duration Quantile')
plt.xticks(index, ('Lower Quantile', 'Middle Quantile', 'Upper Quantile'))
plt.tight_layout()
fig.savefig('./figures/projectDurationByOrgMailDurationQuantiles.png')
plt.show()