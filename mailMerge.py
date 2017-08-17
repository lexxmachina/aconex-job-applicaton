#!/usr/bin/python 

import glob
import pandas as pd
import numpy as np

manifest = pd.read_csv('./manifest.csv', sep=',', names=['projectId','records'], skiprows=[0])
mailTypes = pd.read_csv('./mail_types.csv', sep=',', names=['typeId','typeName'], skiprows=[0])

#----- mailTypes['typeId'] = pd.to_numeric(mailTypes['typeId'], errors='coerce')
#mailTypes['typeId'] = mailTypes['typeId'].astype(str).astype(int)
#print mailTypes.dtypes

mailAll = pd.DataFrame(columns=['projectId', 'correspondenceId', 'sentDate', 'fromOrganizationId', 
            'fromUserId', 'correspondenceTypeId', 'correspondenceTypeName', 'responseRequiredByDate'])

path = './correspondence/' # use your path
allFiles = glob.glob(path + "*.csv")

counter = 0
for file_ in allFiles :
    counter+=1
    print 'files remaining: ' + str(len(allFiles) - counter)

    correspond = pd.read_csv(file_, sep=',', header='infer')
    mail = pd.merge(correspond, mailTypes, how='left', left_on=['correspondenceTypeId'], right_on=['typeId'])
    mail.drop('typeId', axis=1, inplace=True)
    mail.columns = ['projectId', 'correspondenceId', 'sentDate', 'fromOrganizationId', 'fromUserId', 'correspondenceTypeId', 'correspondenceTypeName', 'responseRequiredByDate']
    mailAll = mailAll.append(mail)
    
mailAll_df = pd.DataFrame.from_dict(mailAll)
mailAll_df = mailAll_df[['projectId', 'correspondenceId', 'sentDate', 'fromOrganizationId', 'fromUserId', 'correspondenceTypeId', 'correspondenceTypeName', 'responseRequiredByDate']]
mailAll_df.to_csv('mailAll.csv', sep=',')


