#!/usr/bin/python 

import glob
import pandas as pd
import numpy as np
import json

# Set ipython's max column display
pd.set_option('display.max_columns', 50)

manifest = pd.read_csv('./manifest.csv', sep=',', names=['projectId','records'], skiprows=[0])
mailTypes = pd.read_csv('./mail_types.csv', sep=',', names=['typeId','typeName'], skiprows=[0])

projectStatsAll = []

path = './correspondence/' # use your path
#path = './correspondence_sample/' # use your path
allFiles = glob.glob(path + "*.csv")

counter = 0
for file_ in allFiles :
    print '\nfiles remaining: ' + str(len(allFiles) - counter)
    counter+=1
    
    correspond = pd.read_csv(file_, sep=',', header='infer', keep_default_na=False)

    mail = pd.merge(correspond, mailTypes, how='left', left_on=['correspondenceTypeId'], right_on=['typeId'])
    mail.drop('typeId', axis=1, inplace=True)
    mail.drop('responseRequiredByDate', axis=1, inplace=True)
    mail.columns = ['projectId', 'correspondenceId', 'sentDate', 'fromOrganizationId', 'fromUserId', 'correspondenceTypeId', 'correspondenceTypeName']
    mail['sentDate'] = pd.to_datetime(correspond['sentDate'])
    projectStats = {}
    projectStats['projectId'] = mail['projectId'][0]
    print 'Project ID: ' + str(projectStats['projectId'])
    projectStats['start_ts'] = mail['sentDate'].min()
    projectStats['end_ts'] = mail['sentDate'].max()
    projectStats['projectDuration'] = mail['sentDate'].max() - mail['sentDate'].min()
    projectStats['mailCount'] = mail['correspondenceId'].count()
    projectStats['siteMailCount'] = (mail['correspondenceTypeName'].str.contains(r'Site Instruction')).sum()
    projectStats['varMailCount'] =(mail['correspondenceTypeName'].str.contains(r'Variation')).sum()
    
    # organisation stats
    
    orgIds =  pd.unique(mail[['fromOrganizationId']].values.ravel())
    orgs = []
    for orgId in orgIds :
        org = {}
        org['orgId'] = orgId
        org['orgStart_ts'] = mail['sentDate'].loc[mail['fromOrganizationId'] == orgId].min()
        org['orgEnd_ts'] = mail['sentDate'].loc[mail['fromOrganizationId'] == orgId].max()
        org['orgDuration'] = org['orgEnd_ts'] - org['orgStart_ts']
        org['orgMailCount'] = mail['correspondenceId'].loc[mail['fromOrganizationId'] == orgId].count()
        org['orgSiteMailCount'] = (mail['correspondenceTypeName'].loc[mail['fromOrganizationId'] == orgId].str.contains(r'Site Instruction')).sum()
        org['orgVarMailCount'] = (mail['correspondenceTypeName'].loc[mail['fromOrganizationId'] == orgId].str.contains(r'Variation')).sum()
    #        print org
        orgs.append(org) 
    projectStats['orgs'] = orgs[:]
        
    # user stats
    userIds =  pd.unique(mail[['fromUserId']].values.ravel())
    users = []
    for userId in userIds : #np.nditer(userIds) :
        user = {} 
        user['userId'] = userId
        user['userStart_ts'] = mail['sentDate'].loc[mail['fromUserId'] == userId].min()
        user['userEnd_ts'] = mail['sentDate'].loc[mail['fromUserId'] == userId].max()
        user['userDuration'] = user['userEnd_ts'] - user['userStart_ts']
        user['userMailCount'] = mail['correspondenceId'].loc[mail['fromUserId'] == userId].count()
        user['userSiteMailCount'] = (mail['correspondenceTypeName'].loc[mail['fromUserId'] == userId].str.contains(r'Site Instruction')).sum()
        user['userVarMailCount'] = (mail['correspondenceTypeName'].loc[mail['fromUserId'] == userId].str.contains(r'Variation')).sum()
        users.append(user)
    projectStats['users'] = users[:]
        
    projectStatsAll.append(projectStats) # end of loop

projectStatsAll_df = pd.DataFrame(projectStatsAll)
print projectStatsAll_df
f = open('projectStatsAll.json', 'w')
for row in projectStatsAll_df.iterrows() :
    row[1].to_json(f)
    f.write('\n')
f.close()