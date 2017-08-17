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
    for dic in row['users'] :
        dic['projectId'] = row['projectId']
        dic['projectDuration'] = row['projectDuration']
        dic['mailCount'] = row['mailCount']
        dic['projectStart_ts'] = row['start_ts']
        dic['projectEnd_ts'] = row['end_ts']
        lists.append(dic)

users = pd.DataFrame(lists)

users['userStart_ts'] = pd.to_datetime(users['userStart_ts'], unit='ms')
users['userEnd_ts'] = pd.to_datetime(users['userEnd_ts'], unit='ms')
users['userDuration'] = users['userDuration'] / 1000
users['projectStart_ts'] = pd.to_datetime(users['projectStart_ts'], unit='ms')
users['projectEnd_ts'] = pd.to_datetime(users['projectEnd_ts'], unit='ms')
users['projectDuration'] = users['projectDuration'] / 1000 # scale duration times in microseconds to seconds
projDuration_mean = users['projectDuration'].mean()
users['userDurationNorm'] = users['userDuration'] / projDuration_mean
mailCount_mean = users['mailCount'].mean()
#print mailCount_mean 
users['userMailCountNorm'] = users['userMailCount'] / mailCount_mean
#print users.head(10)

# list site_instructions/variations counts by user Id

userSiteMail = users.groupby('userId').userSiteMailCount.mean().sort_values(ascending=False)
print userSiteMail.head(10)
usersVarMail = users.groupby('userId').userVarMailCount.mean().sort_values(ascending=False)
print usersVarMail.head(10)

userDurationMeans = users.groupby('userId').userDuration.mean().sort_values(ascending=True)
#userDurationMeans.columns=['userId','userDurationMean']
print userDurationMeans.head(10)

#userDurationsQuants = pd.qcut(userDurationMeans, [0, .25, .5, .75, 1.], duplicates='drop') 
userDurationsQuants = []
userDurationsQuants.append(-0.01)
userDurationsQuants.append(userDurationMeans.quantile(0.20))
userDurationsQuants.append(userDurationMeans.quantile(0.60))
userDurationsQuants.append(userDurationMeans.quantile(1.0))
print userDurationsQuants

userDurationMeans = userDurationMeans.to_frame().reset_index()
print (userDurationMeans['userDuration'] / (60 * 60 * 24)).head(20)

userDurationMeans.loc[:, 'bins'] = pd.cut(userDurationMeans['userDuration'], bins=userDurationsQuants, labels=[1,2,3])
cat_columns = userDurationMeans.select_dtypes(['category']).columns

userDurationMeans[cat_columns] = userDurationMeans[cat_columns].apply(lambda x: x.cat.codes)

# split dataframe into training and test sets

df = pd.merge(users[['projectId', 'projectDuration', 'userId']], userDurationMeans, how='left', on=['userId'])
print df.head(10)

#userIdPDMean = pd.DataFrame(df.groupby('userId', as_index=False)['projectDuration'].mean())
#userIdPDMean.rename(columns={'projectDuration':'projectDurationMean'}, inplace=True)
#userIdPDMean['projectDurationMean'] =  userIdPDMean['projectDurationMean'] / (60 * 60 *24)

#print userIdPDMean

train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
test = test.drop_duplicates(subset = 'userId')  #[test['projectId'].isin([topProjIds.index[1]])]
n_bins = train['bins'].nunique()
print train.head(20)

# joining on project duration mean 
#test = pd.merge(test, userIdPDMean, how='inner', on=['userId'])
test['index'] = test.index
#print '\n test set:'
#print test.head(10)

print '\n train: ' 
print train['bins'].head(10)

levels = [x+1 for x in range(n_bins)]
print '\n levels : ' + str(levels)
contrast = Treatment(reference=0).code_without_intercept(levels)
#contrast = Diff().code_without_intercept(levels)
print(contrast.matrix)
#print(contrast.matrix)
print(contrast.matrix[train['bins']-1, :][:20])
print sm.categorical(train['bins'].values)

mod = ols('projectDuration ~ C(bins, Treatment)', data=train)
res = mod.fit()
print(res.summary())

predictions = pd.DataFrame(res.predict(test), columns=['predProjectDuration'])
predictions['projectDurationMean'] = df['projectDuration'].mean()
predictions['index'] = predictions.index
#print predictions
predictions = pd.merge(predictions, test, how='inner', on=['index'])
#predictions['predProjectDuration'] = predictions['predProjectDuration'].astype('timedelta64[s]')
predictions['plotIndex'] = predictions.index

print '\n predictions:'
print predictions.head(20)

# scatter plot 
X = predictions['userId'].head(20).tolist()

print 'X: '
print X
X_bins = predictions['bins'].head(20).tolist()
print X_bins

X_ticks = range(len(X))
X_labels = X
y_pred = (predictions['predProjectDuration'] / (60 * 60 * 24)).head(20).tolist()
print 'y_pred: '
print y_pred
y_act = (predictions['projectDuration'] / (60 * 60 * 24)).head(20).tolist()
print 'y_act: '
print y_act
y_mean = (predictions['projectDurationMean'] / (60 * 60 * 24)).head(20).tolist()
print 'y_mean: '
print y_mean

se = []
se_mean = []

for i in range(len(y_pred)) :  
    se.append((y_pred[i] - y_act[i])**2.)
    se_mean.append((y_mean[i] - y_act[i])**2.)
rmse = np.sqrt(np.mean(se[:]))
print 'rmse : ' + str(rmse)
rmse_mean = np.sqrt(np.mean(se_mean[:]))
print 'rmse_mean: ' + str(rmse_mean)

fig, ax = plt.subplots()

plt.scatter(X_ticks, y_pred, color='b', label='predictions')
plt.scatter(X_ticks, y_act, color='g', label='actual')
plt.scatter(X_ticks, y_mean, color='r', label='mean')

plt.xlabel('User ID')
plt.ylabel('Duration (days)')
plt.title('Project Duration for selected useranisations')
ax.set_xticks(X_ticks)
ax.set_xticklabels(X_labels, rotation=90)
plt.legend()
plt.tight_layout()
fig.savefig('./figures/projectDurationByUserMailDuration.png')
#
plt.show()

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(x, z)
# data to plot

n_groups = n_bins

pd_pred = sorted((predictions['predProjectDuration']/ (60 * 60 * 24)).unique().tolist())
pd_mean = (predictions['projectDurationMean']/ (60 * 60 * 24)).unique().tolist() * n_bins

print pd_pred
print pd_mean

# create plot
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
plt.ylabel('Project Duration')
plt.title('Project Duration by Mail Duration Quantile')
plt.xticks(index, ('Lower Quantile', 'Middle Quantile', 'Upper Quantile'))
 
plt.tight_layout()

fig.savefig('./figures/projectDurationByUserMailDurationQuantiles.png')
plt.show()