import pandas as pd
import numpy as np
import xgboost as xgb
import scipy as sp
def loglossl(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

d2=pd.read_csv('exchange1.csv')
d1=pd.read_csv('notexchange1.csv')

def Get_Cross_Feature(end_days):
    if end_days<31:
        ex0=d1[d1.day==end_days]
        ex1=d2[d2.day==end_days]
        ex=pd.concat([ex0,ex1],axis=0)
    else:
    	 ex=pd.read_csv('test1.csv')
    ex=ex.sort_values('clickTime')
    '''geo特征群（关注）'''
    residence_age1=pd.DataFrame({'residence':d1.residence,'age':d1.age})
    residence_age1=residence_age1.drop_duplicates()
    residence_age2=pd.DataFrame({'residence':d2.residence,'age':d2.age})
    residence_age2=residence_age2.drop_duplicates()
    residence_age=pd.merge(residence_age1,residence_age2,on=['residence','age'],how='outer')
    del residence_age1,residence_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_residence_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_age_exchange_count']=1
        t1=t1.groupby(['residence','age']).agg('sum').reset_index()
        t2=t2.groupby(['residence','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_age_all_count']=t3['preday_'+str(p)+'_residence_age_not_exchange_count']+t3['preday_'+str(p)+'_residence_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_age_exchange_rate']=t3['preday_'+str(p)+'_residence_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_age_all_count'].astype('float')
        residence_age=pd.merge(residence_age,t3,on=['residence','age'],how='outer')
    
    del t1,t2,t3
    residence_age=residence_age.replace(np.nan,0)
    ex=pd.merge(ex,residence_age,on=['residence','age'],how='left')

    residence_gender1=pd.DataFrame({'residence':d1.residence,'gender':d1.gender})
    residence_gender1=residence_gender1.drop_duplicates()
    residence_gender2=pd.DataFrame({'residence':d2.residence,'gender':d2.gender})
    residence_gender2=residence_gender2.drop_duplicates()
    residence_gender=pd.merge(residence_gender1,residence_gender2,on=['residence','gender'],how='outer')
    del residence_gender1,residence_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_residence_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_gender_exchange_count']=1
        t1=t1.groupby(['residence','gender']).agg('sum').reset_index()
        t2=t2.groupby(['residence','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_gender_all_count']=t3['preday_'+str(p)+'_residence_gender_not_exchange_count']+t3['preday_'+str(p)+'_residence_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_gender_exchange_rate']=t3['preday_'+str(p)+'_residence_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_gender_all_count'].astype('float')
        residence_gender=pd.merge(residence_gender,t3,on=['residence','gender'],how='outer')
    
    del t1,t2,t3
    residence_gender=residence_gender.replace(np.nan,0)
    ex=pd.merge(ex,residence_gender,on=['residence','gender'],how='left')

    residence_education1=pd.DataFrame({'residence':d1.residence,'education':d1.education})
    residence_education1=residence_education1.drop_duplicates()
    residence_education2=pd.DataFrame({'residence':d2.residence,'education':d2.education})
    residence_education2=residence_education2.drop_duplicates()
    residence_education=pd.merge(residence_education1,residence_education2,on=['residence','education'],how='outer')
    del residence_education1,residence_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_residence_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_education_exchange_count']=1
        t1=t1.groupby(['residence','education']).agg('sum').reset_index()
        t2=t2.groupby(['residence','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_education_all_count']=t3['preday_'+str(p)+'_residence_education_not_exchange_count']+t3['preday_'+str(p)+'_residence_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_education_exchange_rate']=t3['preday_'+str(p)+'_residence_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_education_all_count'].astype('float')
        residence_education=pd.merge(residence_education,t3,on=['residence','education'],how='outer')
    
    del t1,t2,t3
    residence_education=residence_education.replace(np.nan,0)
    ex=pd.merge(ex,residence_education,on=['residence','education'],how='left')

    residence_haveBaby1=pd.DataFrame({'residence':d1.residence,'haveBaby':d1.haveBaby})
    residence_haveBaby1=residence_haveBaby1.drop_duplicates()
    residence_haveBaby2=pd.DataFrame({'residence':d2.residence,'haveBaby':d2.haveBaby})
    residence_haveBaby2=residence_haveBaby2.drop_duplicates()
    residence_haveBaby=pd.merge(residence_haveBaby1,residence_haveBaby2,on=['residence','haveBaby'],how='outer')
    del residence_haveBaby1,residence_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_residence_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_haveBaby_exchange_count']=1
        t1=t1.groupby(['residence','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['residence','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_haveBaby_all_count']=t3['preday_'+str(p)+'_residence_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_residence_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_residence_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_haveBaby_all_count'].astype('float')
        residence_haveBaby=pd.merge(residence_haveBaby,t3,on=['residence','haveBaby'],how='outer')
    
    del t1,t2,t3
    residence_haveBaby=residence_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,residence_haveBaby,on=['residence','haveBaby'],how='left')

    residence_marriageStatus1=pd.DataFrame({'residence':d1.residence,'marriageStatus':d1.marriageStatus})
    residence_marriageStatus1=residence_marriageStatus1.drop_duplicates()
    residence_marriageStatus2=pd.DataFrame({'residence':d2.residence,'marriageStatus':d2.marriageStatus})
    residence_marriageStatus2=residence_marriageStatus2.drop_duplicates()
    residence_marriageStatus=pd.merge(residence_marriageStatus1,residence_marriageStatus2,on=['residence','marriageStatus'],how='outer')
    del residence_marriageStatus1,residence_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_residence_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_marriageStatus_exchange_count']=1
        t1=t1.groupby(['residence','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['residence','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_marriageStatus_all_count']=t3['preday_'+str(p)+'_residence_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_residence_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_residence_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_marriageStatus_all_count'].astype('float')
        residence_marriageStatus=pd.merge(residence_marriageStatus,t3,on=['residence','marriageStatus'],how='outer')
    
    del t1,t2,t3
    residence_marriageStatus=residence_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,residence_marriageStatus,on=['residence','marriageStatus'],how='left')

    residence_positionID1=pd.DataFrame({'residence':d1.residence,'positionID':d1.positionID})
    residence_positionID1=residence_positionID1.drop_duplicates()
    residence_positionID2=pd.DataFrame({'residence':d2.residence,'positionID':d2.positionID})
    residence_positionID2=residence_positionID2.drop_duplicates()
    residence_positionID=pd.merge(residence_positionID1,residence_positionID2,on=['residence','positionID'],how='outer')
    del residence_positionID1,residence_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_residence_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_positionID_exchange_count']=1
        t1=t1.groupby(['residence','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['residence','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_positionID_all_count']=t3['preday_'+str(p)+'_residence_positionID_not_exchange_count']+t3['preday_'+str(p)+'_residence_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_positionID_exchange_rate']=t3['preday_'+str(p)+'_residence_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_positionID_all_count'].astype('float')
        residence_positionID=pd.merge(residence_positionID,t3,on=['residence','positionID'],how='outer')
    
    del t1,t2,t3
    residence_positionID=residence_positionID.replace(np.nan,0)
    ex=pd.merge(ex,residence_positionID,on=['residence','positionID'],how='left')

    residence_sitesetID1=pd.DataFrame({'residence':d1.residence,'sitesetID':d1.sitesetID})
    residence_sitesetID1=residence_sitesetID1.drop_duplicates()
    residence_sitesetID2=pd.DataFrame({'residence':d2.residence,'sitesetID':d2.sitesetID})
    residence_sitesetID2=residence_sitesetID2.drop_duplicates()
    residence_sitesetID=pd.merge(residence_sitesetID1,residence_sitesetID2,on=['residence','sitesetID'],how='outer')
    del residence_sitesetID1,residence_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_residence_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_sitesetID_exchange_count']=1
        t1=t1.groupby(['residence','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['residence','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_sitesetID_all_count']=t3['preday_'+str(p)+'_residence_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_residence_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_residence_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_sitesetID_all_count'].astype('float')
        residence_sitesetID=pd.merge(residence_sitesetID,t3,on=['residence','sitesetID'],how='outer')
    
    del t1,t2,t3
    residence_sitesetID=residence_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,residence_sitesetID,on=['residence','sitesetID'],how='left')

    residence_positionType1=pd.DataFrame({'residence':d1.residence,'positionType':d1.positionType})
    residence_positionType1=residence_positionType1.drop_duplicates()
    residence_positionType2=pd.DataFrame({'residence':d2.residence,'positionType':d2.positionType})
    residence_positionType2=residence_positionType2.drop_duplicates()
    residence_positionType=pd.merge(residence_positionType1,residence_positionType2,on=['residence','positionType'],how='outer')
    del residence_positionType1,residence_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_residence_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_positionType_exchange_count']=1
        t1=t1.groupby(['residence','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['residence','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_positionType_all_count']=t3['preday_'+str(p)+'_residence_positionType_not_exchange_count']+t3['preday_'+str(p)+'_residence_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_positionType_exchange_rate']=t3['preday_'+str(p)+'_residence_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_positionType_all_count'].astype('float')
        residence_positionType=pd.merge(residence_positionType,t3,on=['residence','positionType'],how='outer')
    
    del t1,t2,t3
    residence_positionType=residence_positionType.replace(np.nan,0)
    ex=pd.merge(ex,residence_positionType,on=['residence','positionType'],how='left')

    residence_connectionType1=pd.DataFrame({'residence':d1.residence,'connectionType':d1.connectionType})
    residence_connectionType1=residence_connectionType1.drop_duplicates()
    residence_connectionType2=pd.DataFrame({'residence':d2.residence,'connectionType':d2.connectionType})
    residence_connectionType2=residence_connectionType2.drop_duplicates()
    residence_connectionType=pd.merge(residence_connectionType1,residence_connectionType2,on=['residence','connectionType'],how='outer')
    del residence_connectionType1,residence_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_residence_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_connectionType_exchange_count']=1
        t1=t1.groupby(['residence','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['residence','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_connectionType_all_count']=t3['preday_'+str(p)+'_residence_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_residence_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_connectionType_exchange_rate']=t3['preday_'+str(p)+'_residence_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_connectionType_all_count'].astype('float')
        residence_connectionType=pd.merge(residence_connectionType,t3,on=['residence','connectionType'],how='outer')
    
    del t1,t2,t3
    residence_connectionType=residence_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,residence_connectionType,on=['residence','connectionType'],how='left')

    residence_telecomsOperator1=pd.DataFrame({'residence':d1.residence,'telecomsOperator':d1.telecomsOperator})
    residence_telecomsOperator1=residence_telecomsOperator1.drop_duplicates()
    residence_telecomsOperator2=pd.DataFrame({'residence':d2.residence,'telecomsOperator':d2.telecomsOperator})
    residence_telecomsOperator2=residence_telecomsOperator2.drop_duplicates()
    residence_telecomsOperator=pd.merge(residence_telecomsOperator1,residence_telecomsOperator2,on=['residence','telecomsOperator'],how='outer')
    del residence_telecomsOperator1,residence_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_residence_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['residence','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['residence','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_telecomsOperator_all_count']=t3['preday_'+str(p)+'_residence_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_residence_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_residence_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_telecomsOperator_all_count'].astype('float')
        residence_telecomsOperator=pd.merge(residence_telecomsOperator,t3,on=['residence','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    residence_telecomsOperator=residence_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,residence_telecomsOperator,on=['residence','telecomsOperator'],how='left')

    residence_appPlatform1=pd.DataFrame({'residence':d1.residence,'appPlatform':d1.appPlatform})
    residence_appPlatform1=residence_appPlatform1.drop_duplicates()
    residence_appPlatform2=pd.DataFrame({'residence':d2.residence,'appPlatform':d2.appPlatform})
    residence_appPlatform2=residence_appPlatform2.drop_duplicates()
    residence_appPlatform=pd.merge(residence_appPlatform1,residence_appPlatform2,on=['residence','appPlatform'],how='outer')
    del residence_appPlatform1,residence_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_residence_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_appPlatform_exchange_count']=1
        t1=t1.groupby(['residence','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['residence','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_appPlatform_all_count']=t3['preday_'+str(p)+'_residence_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_residence_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_residence_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_appPlatform_all_count'].astype('float')
        residence_appPlatform=pd.merge(residence_appPlatform,t3,on=['residence','appPlatform'],how='outer')
    
    del t1,t2,t3
    residence_appPlatform=residence_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,residence_appPlatform,on=['residence','appPlatform'],how='left')

    residence_appCategory1=pd.DataFrame({'residence':d1.residence,'appCategory':d1.appCategory})
    residence_appCategory1=residence_appCategory1.drop_duplicates()
    residence_appCategory2=pd.DataFrame({'residence':d2.residence,'appCategory':d2.appCategory})
    residence_appCategory2=residence_appCategory2.drop_duplicates()
    residence_appCategory=pd.merge(residence_appCategory1,residence_appCategory2,on=['residence','appCategory'],how='outer')
    del residence_appCategory1,residence_appCategory2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'residence':d3.residence,'appCategory':d3.appCategory,'day':d3.day})
        t2=pd.DataFrame({'residence':d4.residence,'appCategory':d4.appCategory,'day':d4.day})
        t1['preday_'+str(p)+'_residence_appCategory_not_exchange_count']=1
        t2['preday_'+str(p)+'_residence_appCategory_exchange_count']=1
        t1=t1.groupby(['residence','appCategory']).agg('sum').reset_index()
        t2=t2.groupby(['residence','appCategory']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['residence','appCategory'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_residence_appCategory_all_count']=t3['preday_'+str(p)+'_residence_appCategory_not_exchange_count']+t3['preday_'+str(p)+'_residence_appCategory_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_residence_appCategory_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_residence_appCategory_exchange_rate']=t3['preday_'+str(p)+'_residence_appCategory_exchange_count'].astype('float')/t3['preday_'+str(p)+'_residence_appCategory_all_count'].astype('float')
        residence_appCategory=pd.merge(residence_appCategory,t3,on=['residence','appCategory'],how='outer')
    
    del t1,t2,t3
    residence_appCategory=residence_appCategory.replace(np.nan,0)
    ex=pd.merge(ex,residence_appCategory,on=['residence','appCategory'],how='left')
    
    '''交叉特征群'''
    '''user_ad特征群/creative'''
    creativeID_age1=pd.DataFrame({'creativeID':d1.creativeID,'age':d1.age})
    creativeID_age1=creativeID_age1.drop_duplicates()
    creativeID_age2=pd.DataFrame({'creativeID':d2.creativeID,'age':d2.age})
    creativeID_age2=creativeID_age2.drop_duplicates()
    creativeID_age=pd.merge(creativeID_age1,creativeID_age2,on=['creativeID','age'],how='outer')
    del creativeID_age1,creativeID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_age_exchange_count']=1
        t1=t1.groupby(['creativeID','age']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_age_all_count']=t3['preday_'+str(p)+'_creativeID_age_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_age_exchange_rate']=t3['preday_'+str(p)+'_creativeID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_age_all_count'].astype('float')
        creativeID_age=pd.merge(creativeID_age,t3,on=['creativeID','age'],how='outer')
    
    del t1,t2,t3
    creativeID_age=creativeID_age.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_age,on=['creativeID','age'],how='left')

    creativeID_gender1=pd.DataFrame({'creativeID':d1.creativeID,'gender':d1.gender})
    creativeID_gender1=creativeID_gender1.drop_duplicates()
    creativeID_gender2=pd.DataFrame({'creativeID':d2.creativeID,'gender':d2.gender})
    creativeID_gender2=creativeID_gender2.drop_duplicates()
    creativeID_gender=pd.merge(creativeID_gender1,creativeID_gender2,on=['creativeID','gender'],how='outer')
    del creativeID_gender1,creativeID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_gender_exchange_count']=1
        t1=t1.groupby(['creativeID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_gender_all_count']=t3['preday_'+str(p)+'_creativeID_gender_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_gender_exchange_rate']=t3['preday_'+str(p)+'_creativeID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_gender_all_count'].astype('float')
        creativeID_gender=pd.merge(creativeID_gender,t3,on=['creativeID','gender'],how='outer')
    
    del t1,t2,t3
    creativeID_gender=creativeID_gender.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_gender,on=['creativeID','gender'],how='left')

    creativeID_education1=pd.DataFrame({'creativeID':d1.creativeID,'education':d1.education})
    creativeID_education1=creativeID_education1.drop_duplicates()
    creativeID_education2=pd.DataFrame({'creativeID':d2.creativeID,'education':d2.education})
    creativeID_education2=creativeID_education2.drop_duplicates()
    creativeID_education=pd.merge(creativeID_education1,creativeID_education2,on=['creativeID','education'],how='outer')
    del creativeID_education1,creativeID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_education_exchange_count']=1
        t1=t1.groupby(['creativeID','education']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_education_all_count']=t3['preday_'+str(p)+'_creativeID_education_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_education_exchange_rate']=t3['preday_'+str(p)+'_creativeID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_education_all_count'].astype('float')
        creativeID_education=pd.merge(creativeID_education,t3,on=['creativeID','education'],how='outer')
    
    del t1,t2,t3
    creativeID_education=creativeID_education.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_education,on=['creativeID','education'],how='left')

    creativeID_haveBaby1=pd.DataFrame({'creativeID':d1.creativeID,'haveBaby':d1.haveBaby})
    creativeID_haveBaby1=creativeID_haveBaby1.drop_duplicates()
    creativeID_haveBaby2=pd.DataFrame({'creativeID':d2.creativeID,'haveBaby':d2.haveBaby})
    creativeID_haveBaby2=creativeID_haveBaby2.drop_duplicates()
    creativeID_haveBaby=pd.merge(creativeID_haveBaby1,creativeID_haveBaby2,on=['creativeID','haveBaby'],how='outer')
    del creativeID_haveBaby1,creativeID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_haveBaby_exchange_count']=1
        t1=t1.groupby(['creativeID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_haveBaby_all_count']=t3['preday_'+str(p)+'_creativeID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_creativeID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_haveBaby_all_count'].astype('float')
        creativeID_haveBaby=pd.merge(creativeID_haveBaby,t3,on=['creativeID','haveBaby'],how='outer')
    
    del t1,t2,t3
    creativeID_haveBaby=creativeID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_haveBaby,on=['creativeID','haveBaby'],how='left')

    creativeID_marriageStatus1=pd.DataFrame({'creativeID':d1.creativeID,'marriageStatus':d1.marriageStatus})
    creativeID_marriageStatus1=creativeID_marriageStatus1.drop_duplicates()
    creativeID_marriageStatus2=pd.DataFrame({'creativeID':d2.creativeID,'marriageStatus':d2.marriageStatus})
    creativeID_marriageStatus2=creativeID_marriageStatus2.drop_duplicates()
    creativeID_marriageStatus=pd.merge(creativeID_marriageStatus1,creativeID_marriageStatus2,on=['creativeID','marriageStatus'],how='outer')
    del creativeID_marriageStatus1,creativeID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['creativeID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_marriageStatus_all_count']=t3['preday_'+str(p)+'_creativeID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_creativeID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_marriageStatus_all_count'].astype('float')
        creativeID_marriageStatus=pd.merge(creativeID_marriageStatus,t3,on=['creativeID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    creativeID_marriageStatus=creativeID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_marriageStatus,on=['creativeID','marriageStatus'],how='left')
    
    '''ad_position特征群/creative'''
    creativeID_positionID1=pd.DataFrame({'creativeID':d1.creativeID,'positionID':d1.positionID})
    creativeID_positionID1=creativeID_positionID1.drop_duplicates()
    creativeID_positionID2=pd.DataFrame({'creativeID':d2.creativeID,'positionID':d2.positionID})
    creativeID_positionID2=creativeID_positionID2.drop_duplicates()
    creativeID_positionID=pd.merge(creativeID_positionID1,creativeID_positionID2,on=['creativeID','positionID'],how='outer')
    del creativeID_positionID1,creativeID_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_positionID_exchange_count']=1
        t1=t1.groupby(['creativeID','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_positionID_all_count']=t3['preday_'+str(p)+'_creativeID_positionID_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_positionID_exchange_rate']=t3['preday_'+str(p)+'_creativeID_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_positionID_all_count'].astype('float')
        creativeID_positionID=pd.merge(creativeID_positionID,t3,on=['creativeID','positionID'],how='outer')
    
    del t1,t2,t3
    creativeID_positionID=creativeID_positionID.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_positionID,on=['creativeID','positionID'],how='left')

    creativeID_sitesetID1=pd.DataFrame({'creativeID':d1.creativeID,'sitesetID':d1.sitesetID})
    creativeID_sitesetID1=creativeID_sitesetID1.drop_duplicates()
    creativeID_sitesetID2=pd.DataFrame({'creativeID':d2.creativeID,'sitesetID':d2.sitesetID})
    creativeID_sitesetID2=creativeID_sitesetID2.drop_duplicates()
    creativeID_sitesetID=pd.merge(creativeID_sitesetID1,creativeID_sitesetID2,on=['creativeID','sitesetID'],how='outer')
    del creativeID_sitesetID1,creativeID_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_sitesetID_exchange_count']=1
        t1=t1.groupby(['creativeID','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_sitesetID_all_count']=t3['preday_'+str(p)+'_creativeID_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_creativeID_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_sitesetID_all_count'].astype('float')
        creativeID_sitesetID=pd.merge(creativeID_sitesetID,t3,on=['creativeID','sitesetID'],how='outer')
    
    del t1,t2,t3
    creativeID_sitesetID=creativeID_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_sitesetID,on=['creativeID','sitesetID'],how='left')

    creativeID_positionType1=pd.DataFrame({'creativeID':d1.creativeID,'positionType':d1.positionType})
    creativeID_positionType1=creativeID_positionType1.drop_duplicates()
    creativeID_positionType2=pd.DataFrame({'creativeID':d2.creativeID,'positionType':d2.positionType})
    creativeID_positionType2=creativeID_positionType2.drop_duplicates()
    creativeID_positionType=pd.merge(creativeID_positionType1,creativeID_positionType2,on=['creativeID','positionType'],how='outer')
    del creativeID_positionType1,creativeID_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_positionType_exchange_count']=1
        t1=t1.groupby(['creativeID','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_positionType_all_count']=t3['preday_'+str(p)+'_creativeID_positionType_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_positionType_exchange_rate']=t3['preday_'+str(p)+'_creativeID_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_positionType_all_count'].astype('float')
        creativeID_positionType=pd.merge(creativeID_positionType,t3,on=['creativeID','positionType'],how='outer')
    
    del t1,t2,t3
    creativeID_positionType=creativeID_positionType.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_positionType,on=['creativeID','positionType'],how='left')

    creativeID_connectionType1=pd.DataFrame({'creativeID':d1.creativeID,'connectionType':d1.connectionType})
    creativeID_connectionType1=creativeID_connectionType1.drop_duplicates()
    creativeID_connectionType2=pd.DataFrame({'creativeID':d2.creativeID,'connectionType':d2.connectionType})
    creativeID_connectionType2=creativeID_connectionType2.drop_duplicates()
    creativeID_connectionType=pd.merge(creativeID_connectionType1,creativeID_connectionType2,on=['creativeID','connectionType'],how='outer')
    del creativeID_connectionType1,creativeID_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_connectionType_exchange_count']=1
        t1=t1.groupby(['creativeID','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_connectionType_all_count']=t3['preday_'+str(p)+'_creativeID_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_connectionType_exchange_rate']=t3['preday_'+str(p)+'_creativeID_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_connectionType_all_count'].astype('float')
        creativeID_connectionType=pd.merge(creativeID_connectionType,t3,on=['creativeID','connectionType'],how='outer')
    
    del t1,t2,t3
    creativeID_connectionType=creativeID_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_connectionType,on=['creativeID','connectionType'],how='left')

    creativeID_telecomsOperator1=pd.DataFrame({'creativeID':d1.creativeID,'telecomsOperator':d1.telecomsOperator})
    creativeID_telecomsOperator1=creativeID_telecomsOperator1.drop_duplicates()
    creativeID_telecomsOperator2=pd.DataFrame({'creativeID':d2.creativeID,'telecomsOperator':d2.telecomsOperator})
    creativeID_telecomsOperator2=creativeID_telecomsOperator2.drop_duplicates()
    creativeID_telecomsOperator=pd.merge(creativeID_telecomsOperator1,creativeID_telecomsOperator2,on=['creativeID','telecomsOperator'],how='outer')
    del creativeID_telecomsOperator1,creativeID_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['creativeID','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_telecomsOperator_all_count']=t3['preday_'+str(p)+'_creativeID_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_creativeID_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_telecomsOperator_all_count'].astype('float')
        creativeID_telecomsOperator=pd.merge(creativeID_telecomsOperator,t3,on=['creativeID','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    creativeID_telecomsOperator=creativeID_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_telecomsOperator,on=['creativeID','telecomsOperator'],how='left')

    creativeID_appPlatform1=pd.DataFrame({'creativeID':d1.creativeID,'appPlatform':d1.appPlatform})
    creativeID_appPlatform1=creativeID_appPlatform1.drop_duplicates()
    creativeID_appPlatform2=pd.DataFrame({'creativeID':d2.creativeID,'appPlatform':d2.appPlatform})
    creativeID_appPlatform2=creativeID_appPlatform2.drop_duplicates()
    creativeID_appPlatform=pd.merge(creativeID_appPlatform1,creativeID_appPlatform2,on=['creativeID','appPlatform'],how='outer')
    del creativeID_appPlatform1,creativeID_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'creativeID':d3.creativeID,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'creativeID':d4.creativeID,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_creativeID_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_creativeID_appPlatform_exchange_count']=1
        t1=t1.groupby(['creativeID','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['creativeID','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['creativeID','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_creativeID_appPlatform_all_count']=t3['preday_'+str(p)+'_creativeID_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_creativeID_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_creativeID_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_creativeID_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_creativeID_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_creativeID_appPlatform_all_count'].astype('float')
        creativeID_appPlatform=pd.merge(creativeID_appPlatform,t3,on=['creativeID','appPlatform'],how='outer')
    
    del t1,t2,t3
    creativeID_appPlatform=creativeID_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,creativeID_appPlatform,on=['creativeID','appPlatform'],how='left')

    '''user_ad特征群/advertiser'''
    advertiserID_age1=pd.DataFrame({'advertiserID':d1.advertiserID,'age':d1.age})
    advertiserID_age1=advertiserID_age1.drop_duplicates()
    advertiserID_age2=pd.DataFrame({'advertiserID':d2.advertiserID,'age':d2.age})
    advertiserID_age2=advertiserID_age2.drop_duplicates()
    advertiserID_age=pd.merge(advertiserID_age1,advertiserID_age2,on=['advertiserID','age'],how='outer')
    del advertiserID_age1,advertiserID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_age_exchange_count']=1
        t1=t1.groupby(['advertiserID','age']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_age_all_count']=t3['preday_'+str(p)+'_advertiserID_age_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_age_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_age_all_count'].astype('float')
        advertiserID_age=pd.merge(advertiserID_age,t3,on=['advertiserID','age'],how='outer')
    
    del t1,t2,t3
    advertiserID_age=advertiserID_age.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_age,on=['advertiserID','age'],how='left')

    advertiserID_gender1=pd.DataFrame({'advertiserID':d1.advertiserID,'gender':d1.gender})
    advertiserID_gender1=advertiserID_gender1.drop_duplicates()
    advertiserID_gender2=pd.DataFrame({'advertiserID':d2.advertiserID,'gender':d2.gender})
    advertiserID_gender2=advertiserID_gender2.drop_duplicates()
    advertiserID_gender=pd.merge(advertiserID_gender1,advertiserID_gender2,on=['advertiserID','gender'],how='outer')
    del advertiserID_gender1,advertiserID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_gender_exchange_count']=1
        t1=t1.groupby(['advertiserID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_gender_all_count']=t3['preday_'+str(p)+'_advertiserID_gender_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_gender_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_gender_all_count'].astype('float')
        advertiserID_gender=pd.merge(advertiserID_gender,t3,on=['advertiserID','gender'],how='outer')
    
    del t1,t2,t3
    advertiserID_gender=advertiserID_gender.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_gender,on=['advertiserID','gender'],how='left')

    advertiserID_education1=pd.DataFrame({'advertiserID':d1.advertiserID,'education':d1.education})
    advertiserID_education1=advertiserID_education1.drop_duplicates()
    advertiserID_education2=pd.DataFrame({'advertiserID':d2.advertiserID,'education':d2.education})
    advertiserID_education2=advertiserID_education2.drop_duplicates()
    advertiserID_education=pd.merge(advertiserID_education1,advertiserID_education2,on=['advertiserID','education'],how='outer')
    del advertiserID_education1,advertiserID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_education_exchange_count']=1
        t1=t1.groupby(['advertiserID','education']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_education_all_count']=t3['preday_'+str(p)+'_advertiserID_education_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_education_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_education_all_count'].astype('float')
        advertiserID_education=pd.merge(advertiserID_education,t3,on=['advertiserID','education'],how='outer')
    
    del t1,t2,t3
    advertiserID_education=advertiserID_education.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_education,on=['advertiserID','education'],how='left')

    advertiserID_haveBaby1=pd.DataFrame({'advertiserID':d1.advertiserID,'haveBaby':d1.haveBaby})
    advertiserID_haveBaby1=advertiserID_haveBaby1.drop_duplicates()
    advertiserID_haveBaby2=pd.DataFrame({'advertiserID':d2.advertiserID,'haveBaby':d2.haveBaby})
    advertiserID_haveBaby2=advertiserID_haveBaby2.drop_duplicates()
    advertiserID_haveBaby=pd.merge(advertiserID_haveBaby1,advertiserID_haveBaby2,on=['advertiserID','haveBaby'],how='outer')
    del advertiserID_haveBaby1,advertiserID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_haveBaby_exchange_count']=1
        t1=t1.groupby(['advertiserID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_haveBaby_all_count']=t3['preday_'+str(p)+'_advertiserID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_haveBaby_all_count'].astype('float')
        advertiserID_haveBaby=pd.merge(advertiserID_haveBaby,t3,on=['advertiserID','haveBaby'],how='outer')
    
    del t1,t2,t3
    advertiserID_haveBaby=advertiserID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_haveBaby,on=['advertiserID','haveBaby'],how='left')

    advertiserID_marriageStatus1=pd.DataFrame({'advertiserID':d1.advertiserID,'marriageStatus':d1.marriageStatus})
    advertiserID_marriageStatus1=advertiserID_marriageStatus1.drop_duplicates()
    advertiserID_marriageStatus2=pd.DataFrame({'advertiserID':d2.advertiserID,'marriageStatus':d2.marriageStatus})
    advertiserID_marriageStatus2=advertiserID_marriageStatus2.drop_duplicates()
    advertiserID_marriageStatus=pd.merge(advertiserID_marriageStatus1,advertiserID_marriageStatus2,on=['advertiserID','marriageStatus'],how='outer')
    del advertiserID_marriageStatus1,advertiserID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['advertiserID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_marriageStatus_all_count']=t3['preday_'+str(p)+'_advertiserID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_marriageStatus_all_count'].astype('float')
        advertiserID_marriageStatus=pd.merge(advertiserID_marriageStatus,t3,on=['advertiserID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    advertiserID_marriageStatus=advertiserID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_marriageStatus,on=['advertiserID','marriageStatus'],how='left')
    
    '''ad_position特征群/advertiser'''
    advertiserID_positionID1=pd.DataFrame({'advertiserID':d1.advertiserID,'positionID':d1.positionID})
    advertiserID_positionID1=advertiserID_positionID1.drop_duplicates()
    advertiserID_positionID2=pd.DataFrame({'advertiserID':d2.advertiserID,'positionID':d2.positionID})
    advertiserID_positionID2=advertiserID_positionID2.drop_duplicates()
    advertiserID_positionID=pd.merge(advertiserID_positionID1,advertiserID_positionID2,on=['advertiserID','positionID'],how='outer')
    del advertiserID_positionID1,advertiserID_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_positionID_exchange_count']=1
        t1=t1.groupby(['advertiserID','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_positionID_all_count']=t3['preday_'+str(p)+'_advertiserID_positionID_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_positionID_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_positionID_all_count'].astype('float')
        advertiserID_positionID=pd.merge(advertiserID_positionID,t3,on=['advertiserID','positionID'],how='outer')
    
    del t1,t2,t3
    advertiserID_positionID=advertiserID_positionID.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_positionID,on=['advertiserID','positionID'],how='left')

    advertiserID_sitesetID1=pd.DataFrame({'advertiserID':d1.advertiserID,'sitesetID':d1.sitesetID})
    advertiserID_sitesetID1=advertiserID_sitesetID1.drop_duplicates()
    advertiserID_sitesetID2=pd.DataFrame({'advertiserID':d2.advertiserID,'sitesetID':d2.sitesetID})
    advertiserID_sitesetID2=advertiserID_sitesetID2.drop_duplicates()
    advertiserID_sitesetID=pd.merge(advertiserID_sitesetID1,advertiserID_sitesetID2,on=['advertiserID','sitesetID'],how='outer')
    del advertiserID_sitesetID1,advertiserID_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_sitesetID_exchange_count']=1
        t1=t1.groupby(['advertiserID','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_sitesetID_all_count']=t3['preday_'+str(p)+'_advertiserID_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_sitesetID_all_count'].astype('float')
        advertiserID_sitesetID=pd.merge(advertiserID_sitesetID,t3,on=['advertiserID','sitesetID'],how='outer')
    
    del t1,t2,t3
    advertiserID_sitesetID=advertiserID_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_sitesetID,on=['advertiserID','sitesetID'],how='left')

    advertiserID_positionType1=pd.DataFrame({'advertiserID':d1.advertiserID,'positionType':d1.positionType})
    advertiserID_positionType1=advertiserID_positionType1.drop_duplicates()
    advertiserID_positionType2=pd.DataFrame({'advertiserID':d2.advertiserID,'positionType':d2.positionType})
    advertiserID_positionType2=advertiserID_positionType2.drop_duplicates()
    advertiserID_positionType=pd.merge(advertiserID_positionType1,advertiserID_positionType2,on=['advertiserID','positionType'],how='outer')
    del advertiserID_positionType1,advertiserID_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_positionType_exchange_count']=1
        t1=t1.groupby(['advertiserID','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_positionType_all_count']=t3['preday_'+str(p)+'_advertiserID_positionType_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_positionType_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_positionType_all_count'].astype('float')
        advertiserID_positionType=pd.merge(advertiserID_positionType,t3,on=['advertiserID','positionType'],how='outer')
    
    del t1,t2,t3
    advertiserID_positionType=advertiserID_positionType.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_positionType,on=['advertiserID','positionType'],how='left')

    advertiserID_connectionType1=pd.DataFrame({'advertiserID':d1.advertiserID,'connectionType':d1.connectionType})
    advertiserID_connectionType1=advertiserID_connectionType1.drop_duplicates()
    advertiserID_connectionType2=pd.DataFrame({'advertiserID':d2.advertiserID,'connectionType':d2.connectionType})
    advertiserID_connectionType2=advertiserID_connectionType2.drop_duplicates()
    advertiserID_connectionType=pd.merge(advertiserID_connectionType1,advertiserID_connectionType2,on=['advertiserID','connectionType'],how='outer')
    del advertiserID_connectionType1,advertiserID_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_connectionType_exchange_count']=1
        t1=t1.groupby(['advertiserID','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_connectionType_all_count']=t3['preday_'+str(p)+'_advertiserID_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_connectionType_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_connectionType_all_count'].astype('float')
        advertiserID_connectionType=pd.merge(advertiserID_connectionType,t3,on=['advertiserID','connectionType'],how='outer')
    
    del t1,t2,t3
    advertiserID_connectionType=advertiserID_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_connectionType,on=['advertiserID','connectionType'],how='left')

    advertiserID_telecomsOperator1=pd.DataFrame({'advertiserID':d1.advertiserID,'telecomsOperator':d1.telecomsOperator})
    advertiserID_telecomsOperator1=advertiserID_telecomsOperator1.drop_duplicates()
    advertiserID_telecomsOperator2=pd.DataFrame({'advertiserID':d2.advertiserID,'telecomsOperator':d2.telecomsOperator})
    advertiserID_telecomsOperator2=advertiserID_telecomsOperator2.drop_duplicates()
    advertiserID_telecomsOperator=pd.merge(advertiserID_telecomsOperator1,advertiserID_telecomsOperator2,on=['advertiserID','telecomsOperator'],how='outer')
    del advertiserID_telecomsOperator1,advertiserID_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['advertiserID','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_telecomsOperator_all_count']=t3['preday_'+str(p)+'_advertiserID_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_telecomsOperator_all_count'].astype('float')
        advertiserID_telecomsOperator=pd.merge(advertiserID_telecomsOperator,t3,on=['advertiserID','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    advertiserID_telecomsOperator=advertiserID_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_telecomsOperator,on=['advertiserID','telecomsOperator'],how='left')

    advertiserID_appPlatform1=pd.DataFrame({'advertiserID':d1.advertiserID,'appPlatform':d1.appPlatform})
    advertiserID_appPlatform1=advertiserID_appPlatform1.drop_duplicates()
    advertiserID_appPlatform2=pd.DataFrame({'advertiserID':d2.advertiserID,'appPlatform':d2.appPlatform})
    advertiserID_appPlatform2=advertiserID_appPlatform2.drop_duplicates()
    advertiserID_appPlatform=pd.merge(advertiserID_appPlatform1,advertiserID_appPlatform2,on=['advertiserID','appPlatform'],how='outer')
    del advertiserID_appPlatform1,advertiserID_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'advertiserID':d3.advertiserID,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'advertiserID':d4.advertiserID,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_advertiserID_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_advertiserID_appPlatform_exchange_count']=1
        t1=t1.groupby(['advertiserID','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['advertiserID','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['advertiserID','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_advertiserID_appPlatform_all_count']=t3['preday_'+str(p)+'_advertiserID_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_advertiserID_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_advertiserID_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_advertiserID_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_advertiserID_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_advertiserID_appPlatform_all_count'].astype('float')
        advertiserID_appPlatform=pd.merge(advertiserID_appPlatform,t3,on=['advertiserID','appPlatform'],how='outer')
    
    del t1,t2,t3
    advertiserID_appPlatform=advertiserID_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,advertiserID_appPlatform,on=['advertiserID','appPlatform'],how='left')
    
    '''user_ad特征群/camgaign'''
    camgaignID_age1=pd.DataFrame({'camgaignID':d1.camgaignID,'age':d1.age})
    camgaignID_age1=camgaignID_age1.drop_duplicates()
    camgaignID_age2=pd.DataFrame({'camgaignID':d2.camgaignID,'age':d2.age})
    camgaignID_age2=camgaignID_age2.drop_duplicates()
    camgaignID_age=pd.merge(camgaignID_age1,camgaignID_age2,on=['camgaignID','age'],how='outer')
    del camgaignID_age1,camgaignID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_age_exchange_count']=1
        t1=t1.groupby(['camgaignID','age']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_age_all_count']=t3['preday_'+str(p)+'_camgaignID_age_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_age_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_age_all_count'].astype('float')
        camgaignID_age=pd.merge(camgaignID_age,t3,on=['camgaignID','age'],how='outer')
    
    del t1,t2,t3
    camgaignID_age=camgaignID_age.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_age,on=['camgaignID','age'],how='left')

    camgaignID_gender1=pd.DataFrame({'camgaignID':d1.camgaignID,'gender':d1.gender})
    camgaignID_gender1=camgaignID_gender1.drop_duplicates()
    camgaignID_gender2=pd.DataFrame({'camgaignID':d2.camgaignID,'gender':d2.gender})
    camgaignID_gender2=camgaignID_gender2.drop_duplicates()
    camgaignID_gender=pd.merge(camgaignID_gender1,camgaignID_gender2,on=['camgaignID','gender'],how='outer')
    del camgaignID_gender1,camgaignID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_gender_exchange_count']=1
        t1=t1.groupby(['camgaignID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_gender_all_count']=t3['preday_'+str(p)+'_camgaignID_gender_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_gender_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_gender_all_count'].astype('float')
        camgaignID_gender=pd.merge(camgaignID_gender,t3,on=['camgaignID','gender'],how='outer')
    
    del t1,t2,t3
    camgaignID_gender=camgaignID_gender.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_gender,on=['camgaignID','gender'],how='left')

    camgaignID_education1=pd.DataFrame({'camgaignID':d1.camgaignID,'education':d1.education})
    camgaignID_education1=camgaignID_education1.drop_duplicates()
    camgaignID_education2=pd.DataFrame({'camgaignID':d2.camgaignID,'education':d2.education})
    camgaignID_education2=camgaignID_education2.drop_duplicates()
    camgaignID_education=pd.merge(camgaignID_education1,camgaignID_education2,on=['camgaignID','education'],how='outer')
    del camgaignID_education1,camgaignID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_education_exchange_count']=1
        t1=t1.groupby(['camgaignID','education']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_education_all_count']=t3['preday_'+str(p)+'_camgaignID_education_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_education_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_education_all_count'].astype('float')
        camgaignID_education=pd.merge(camgaignID_education,t3,on=['camgaignID','education'],how='outer')
    
    del t1,t2,t3
    camgaignID_education=camgaignID_education.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_education,on=['camgaignID','education'],how='left')

    camgaignID_haveBaby1=pd.DataFrame({'camgaignID':d1.camgaignID,'haveBaby':d1.haveBaby})
    camgaignID_haveBaby1=camgaignID_haveBaby1.drop_duplicates()
    camgaignID_haveBaby2=pd.DataFrame({'camgaignID':d2.camgaignID,'haveBaby':d2.haveBaby})
    camgaignID_haveBaby2=camgaignID_haveBaby2.drop_duplicates()
    camgaignID_haveBaby=pd.merge(camgaignID_haveBaby1,camgaignID_haveBaby2,on=['camgaignID','haveBaby'],how='outer')
    del camgaignID_haveBaby1,camgaignID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_haveBaby_exchange_count']=1
        t1=t1.groupby(['camgaignID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_haveBaby_all_count']=t3['preday_'+str(p)+'_camgaignID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_haveBaby_all_count'].astype('float')
        camgaignID_haveBaby=pd.merge(camgaignID_haveBaby,t3,on=['camgaignID','haveBaby'],how='outer')
    
    del t1,t2,t3
    camgaignID_haveBaby=camgaignID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_haveBaby,on=['camgaignID','haveBaby'],how='left')

    camgaignID_marriageStatus1=pd.DataFrame({'camgaignID':d1.camgaignID,'marriageStatus':d1.marriageStatus})
    camgaignID_marriageStatus1=camgaignID_marriageStatus1.drop_duplicates()
    camgaignID_marriageStatus2=pd.DataFrame({'camgaignID':d2.camgaignID,'marriageStatus':d2.marriageStatus})
    camgaignID_marriageStatus2=camgaignID_marriageStatus2.drop_duplicates()
    camgaignID_marriageStatus=pd.merge(camgaignID_marriageStatus1,camgaignID_marriageStatus2,on=['camgaignID','marriageStatus'],how='outer')
    del camgaignID_marriageStatus1,camgaignID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['camgaignID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_marriageStatus_all_count']=t3['preday_'+str(p)+'_camgaignID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_marriageStatus_all_count'].astype('float')
        camgaignID_marriageStatus=pd.merge(camgaignID_marriageStatus,t3,on=['camgaignID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    camgaignID_marriageStatus=camgaignID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_marriageStatus,on=['camgaignID','marriageStatus'],how='left')
    
    '''ad_position特征群/camgaign'''
    camgaignID_positionID1=pd.DataFrame({'camgaignID':d1.camgaignID,'positionID':d1.positionID})
    camgaignID_positionID1=camgaignID_positionID1.drop_duplicates()
    camgaignID_positionID2=pd.DataFrame({'camgaignID':d2.camgaignID,'positionID':d2.positionID})
    camgaignID_positionID2=camgaignID_positionID2.drop_duplicates()
    camgaignID_positionID=pd.merge(camgaignID_positionID1,camgaignID_positionID2,on=['camgaignID','positionID'],how='outer')
    del camgaignID_positionID1,camgaignID_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_positionID_exchange_count']=1
        t1=t1.groupby(['camgaignID','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_positionID_all_count']=t3['preday_'+str(p)+'_camgaignID_positionID_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_positionID_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_positionID_all_count'].astype('float')
        camgaignID_positionID=pd.merge(camgaignID_positionID,t3,on=['camgaignID','positionID'],how='outer')
    
    del t1,t2,t3
    camgaignID_positionID=camgaignID_positionID.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_positionID,on=['camgaignID','positionID'],how='left')

    camgaignID_sitesetID1=pd.DataFrame({'camgaignID':d1.camgaignID,'sitesetID':d1.sitesetID})
    camgaignID_sitesetID1=camgaignID_sitesetID1.drop_duplicates()
    camgaignID_sitesetID2=pd.DataFrame({'camgaignID':d2.camgaignID,'sitesetID':d2.sitesetID})
    camgaignID_sitesetID2=camgaignID_sitesetID2.drop_duplicates()
    camgaignID_sitesetID=pd.merge(camgaignID_sitesetID1,camgaignID_sitesetID2,on=['camgaignID','sitesetID'],how='outer')
    del camgaignID_sitesetID1,camgaignID_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_sitesetID_exchange_count']=1
        t1=t1.groupby(['camgaignID','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_sitesetID_all_count']=t3['preday_'+str(p)+'_camgaignID_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_sitesetID_all_count'].astype('float')
        camgaignID_sitesetID=pd.merge(camgaignID_sitesetID,t3,on=['camgaignID','sitesetID'],how='outer')
    
    del t1,t2,t3
    camgaignID_sitesetID=camgaignID_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_sitesetID,on=['camgaignID','sitesetID'],how='left')

    camgaignID_positionType1=pd.DataFrame({'camgaignID':d1.camgaignID,'positionType':d1.positionType})
    camgaignID_positionType1=camgaignID_positionType1.drop_duplicates()
    camgaignID_positionType2=pd.DataFrame({'camgaignID':d2.camgaignID,'positionType':d2.positionType})
    camgaignID_positionType2=camgaignID_positionType2.drop_duplicates()
    camgaignID_positionType=pd.merge(camgaignID_positionType1,camgaignID_positionType2,on=['camgaignID','positionType'],how='outer')
    del camgaignID_positionType1,camgaignID_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_positionType_exchange_count']=1
        t1=t1.groupby(['camgaignID','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_positionType_all_count']=t3['preday_'+str(p)+'_camgaignID_positionType_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_positionType_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_positionType_all_count'].astype('float')
        camgaignID_positionType=pd.merge(camgaignID_positionType,t3,on=['camgaignID','positionType'],how='outer')
    
    del t1,t2,t3
    camgaignID_positionType=camgaignID_positionType.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_positionType,on=['camgaignID','positionType'],how='left')

    camgaignID_connectionType1=pd.DataFrame({'camgaignID':d1.camgaignID,'connectionType':d1.connectionType})
    camgaignID_connectionType1=camgaignID_connectionType1.drop_duplicates()
    camgaignID_connectionType2=pd.DataFrame({'camgaignID':d2.camgaignID,'connectionType':d2.connectionType})
    camgaignID_connectionType2=camgaignID_connectionType2.drop_duplicates()
    camgaignID_connectionType=pd.merge(camgaignID_connectionType1,camgaignID_connectionType2,on=['camgaignID','connectionType'],how='outer')
    del camgaignID_connectionType1,camgaignID_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_connectionType_exchange_count']=1
        t1=t1.groupby(['camgaignID','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_connectionType_all_count']=t3['preday_'+str(p)+'_camgaignID_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_connectionType_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_connectionType_all_count'].astype('float')
        camgaignID_connectionType=pd.merge(camgaignID_connectionType,t3,on=['camgaignID','connectionType'],how='outer')
    
    del t1,t2,t3
    camgaignID_connectionType=camgaignID_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_connectionType,on=['camgaignID','connectionType'],how='left')

    camgaignID_telecomsOperator1=pd.DataFrame({'camgaignID':d1.camgaignID,'telecomsOperator':d1.telecomsOperator})
    camgaignID_telecomsOperator1=camgaignID_telecomsOperator1.drop_duplicates()
    camgaignID_telecomsOperator2=pd.DataFrame({'camgaignID':d2.camgaignID,'telecomsOperator':d2.telecomsOperator})
    camgaignID_telecomsOperator2=camgaignID_telecomsOperator2.drop_duplicates()
    camgaignID_telecomsOperator=pd.merge(camgaignID_telecomsOperator1,camgaignID_telecomsOperator2,on=['camgaignID','telecomsOperator'],how='outer')
    del camgaignID_telecomsOperator1,camgaignID_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['camgaignID','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_telecomsOperator_all_count']=t3['preday_'+str(p)+'_camgaignID_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_telecomsOperator_all_count'].astype('float')
        camgaignID_telecomsOperator=pd.merge(camgaignID_telecomsOperator,t3,on=['camgaignID','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    camgaignID_telecomsOperator=camgaignID_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_telecomsOperator,on=['camgaignID','telecomsOperator'],how='left')

    camgaignID_appPlatform1=pd.DataFrame({'camgaignID':d1.camgaignID,'appPlatform':d1.appPlatform})
    camgaignID_appPlatform1=camgaignID_appPlatform1.drop_duplicates()
    camgaignID_appPlatform2=pd.DataFrame({'camgaignID':d2.camgaignID,'appPlatform':d2.appPlatform})
    camgaignID_appPlatform2=camgaignID_appPlatform2.drop_duplicates()
    camgaignID_appPlatform=pd.merge(camgaignID_appPlatform1,camgaignID_appPlatform2,on=['camgaignID','appPlatform'],how='outer')
    del camgaignID_appPlatform1,camgaignID_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'camgaignID':d3.camgaignID,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'camgaignID':d4.camgaignID,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_camgaignID_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_camgaignID_appPlatform_exchange_count']=1
        t1=t1.groupby(['camgaignID','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['camgaignID','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['camgaignID','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_camgaignID_appPlatform_all_count']=t3['preday_'+str(p)+'_camgaignID_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_camgaignID_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_camgaignID_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_camgaignID_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_camgaignID_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_camgaignID_appPlatform_all_count'].astype('float')
        camgaignID_appPlatform=pd.merge(camgaignID_appPlatform,t3,on=['camgaignID','appPlatform'],how='outer')
    
    del t1,t2,t3
    camgaignID_appPlatform=camgaignID_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,camgaignID_appPlatform,on=['camgaignID','appPlatform'],how='left')


    '''user_ad特征群/app'''
    appID_age1=pd.DataFrame({'appID':d1.appID,'age':d1.age})
    appID_age1=appID_age1.drop_duplicates()
    appID_age2=pd.DataFrame({'appID':d2.appID,'age':d2.age})
    appID_age2=appID_age2.drop_duplicates()
    appID_age=pd.merge(appID_age1,appID_age2,on=['appID','age'],how='outer')
    del appID_age1,appID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_appID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_age_exchange_count']=1
        t1=t1.groupby(['appID','age']).agg('sum').reset_index()
        t2=t2.groupby(['appID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_age_all_count']=t3['preday_'+str(p)+'_appID_age_not_exchange_count']+t3['preday_'+str(p)+'_appID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_age_exchange_rate']=t3['preday_'+str(p)+'_appID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_age_all_count'].astype('float')
        appID_age=pd.merge(appID_age,t3,on=['appID','age'],how='outer')
    
    del t1,t2,t3
    appID_age=appID_age.replace(np.nan,0)
    ex=pd.merge(ex,appID_age,on=['appID','age'],how='left')

    appID_gender1=pd.DataFrame({'appID':d1.appID,'gender':d1.gender})
    appID_gender1=appID_gender1.drop_duplicates()
    appID_gender2=pd.DataFrame({'appID':d2.appID,'gender':d2.gender})
    appID_gender2=appID_gender2.drop_duplicates()
    appID_gender=pd.merge(appID_gender1,appID_gender2,on=['appID','gender'],how='outer')
    del appID_gender1,appID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_appID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_gender_exchange_count']=1
        t1=t1.groupby(['appID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['appID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_gender_all_count']=t3['preday_'+str(p)+'_appID_gender_not_exchange_count']+t3['preday_'+str(p)+'_appID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_gender_exchange_rate']=t3['preday_'+str(p)+'_appID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_gender_all_count'].astype('float')
        appID_gender=pd.merge(appID_gender,t3,on=['appID','gender'],how='outer')
    
    del t1,t2,t3
    appID_gender=appID_gender.replace(np.nan,0)
    ex=pd.merge(ex,appID_gender,on=['appID','gender'],how='left')

    appID_education1=pd.DataFrame({'appID':d1.appID,'education':d1.education})
    appID_education1=appID_education1.drop_duplicates()
    appID_education2=pd.DataFrame({'appID':d2.appID,'education':d2.education})
    appID_education2=appID_education2.drop_duplicates()
    appID_education=pd.merge(appID_education1,appID_education2,on=['appID','education'],how='outer')
    del appID_education1,appID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_appID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_education_exchange_count']=1
        t1=t1.groupby(['appID','education']).agg('sum').reset_index()
        t2=t2.groupby(['appID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_education_all_count']=t3['preday_'+str(p)+'_appID_education_not_exchange_count']+t3['preday_'+str(p)+'_appID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_education_exchange_rate']=t3['preday_'+str(p)+'_appID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_education_all_count'].astype('float')
        appID_education=pd.merge(appID_education,t3,on=['appID','education'],how='outer')
    
    del t1,t2,t3
    appID_education=appID_education.replace(np.nan,0)
    ex=pd.merge(ex,appID_education,on=['appID','education'],how='left')

    appID_haveBaby1=pd.DataFrame({'appID':d1.appID,'haveBaby':d1.haveBaby})
    appID_haveBaby1=appID_haveBaby1.drop_duplicates()
    appID_haveBaby2=pd.DataFrame({'appID':d2.appID,'haveBaby':d2.haveBaby})
    appID_haveBaby2=appID_haveBaby2.drop_duplicates()
    appID_haveBaby=pd.merge(appID_haveBaby1,appID_haveBaby2,on=['appID','haveBaby'],how='outer')
    del appID_haveBaby1,appID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_appID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_haveBaby_exchange_count']=1
        t1=t1.groupby(['appID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['appID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_haveBaby_all_count']=t3['preday_'+str(p)+'_appID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_appID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_appID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_haveBaby_all_count'].astype('float')
        appID_haveBaby=pd.merge(appID_haveBaby,t3,on=['appID','haveBaby'],how='outer')
    
    del t1,t2,t3
    appID_haveBaby=appID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,appID_haveBaby,on=['appID','haveBaby'],how='left')

    appID_marriageStatus1=pd.DataFrame({'appID':d1.appID,'marriageStatus':d1.marriageStatus})
    appID_marriageStatus1=appID_marriageStatus1.drop_duplicates()
    appID_marriageStatus2=pd.DataFrame({'appID':d2.appID,'marriageStatus':d2.marriageStatus})
    appID_marriageStatus2=appID_marriageStatus2.drop_duplicates()
    appID_marriageStatus=pd.merge(appID_marriageStatus1,appID_marriageStatus2,on=['appID','marriageStatus'],how='outer')
    del appID_marriageStatus1,appID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_appID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['appID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['appID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_marriageStatus_all_count']=t3['preday_'+str(p)+'_appID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_appID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_appID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_marriageStatus_all_count'].astype('float')
        appID_marriageStatus=pd.merge(appID_marriageStatus,t3,on=['appID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    appID_marriageStatus=appID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,appID_marriageStatus,on=['appID','marriageStatus'],how='left')
    
    '''ad_position特征群/app'''
    appID_positionID1=pd.DataFrame({'appID':d1.appID,'positionID':d1.positionID})
    appID_positionID1=appID_positionID1.drop_duplicates()
    appID_positionID2=pd.DataFrame({'appID':d2.appID,'positionID':d2.positionID})
    appID_positionID2=appID_positionID2.drop_duplicates()
    appID_positionID=pd.merge(appID_positionID1,appID_positionID2,on=['appID','positionID'],how='outer')
    del appID_positionID1,appID_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_appID_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_positionID_exchange_count']=1
        t1=t1.groupby(['appID','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['appID','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_positionID_all_count']=t3['preday_'+str(p)+'_appID_positionID_not_exchange_count']+t3['preday_'+str(p)+'_appID_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_positionID_exchange_rate']=t3['preday_'+str(p)+'_appID_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_positionID_all_count'].astype('float')
        appID_positionID=pd.merge(appID_positionID,t3,on=['appID','positionID'],how='outer')
    
    del t1,t2,t3
    appID_positionID=appID_positionID.replace(np.nan,0)
    ex=pd.merge(ex,appID_positionID,on=['appID','positionID'],how='left')

    appID_sitesetID1=pd.DataFrame({'appID':d1.appID,'sitesetID':d1.sitesetID})
    appID_sitesetID1=appID_sitesetID1.drop_duplicates()
    appID_sitesetID2=pd.DataFrame({'appID':d2.appID,'sitesetID':d2.sitesetID})
    appID_sitesetID2=appID_sitesetID2.drop_duplicates()
    appID_sitesetID=pd.merge(appID_sitesetID1,appID_sitesetID2,on=['appID','sitesetID'],how='outer')
    del appID_sitesetID1,appID_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_appID_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_sitesetID_exchange_count']=1
        t1=t1.groupby(['appID','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['appID','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_sitesetID_all_count']=t3['preday_'+str(p)+'_appID_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_appID_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_appID_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_sitesetID_all_count'].astype('float')
        appID_sitesetID=pd.merge(appID_sitesetID,t3,on=['appID','sitesetID'],how='outer')
    
    del t1,t2,t3
    appID_sitesetID=appID_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,appID_sitesetID,on=['appID','sitesetID'],how='left')

    appID_positionType1=pd.DataFrame({'appID':d1.appID,'positionType':d1.positionType})
    appID_positionType1=appID_positionType1.drop_duplicates()
    appID_positionType2=pd.DataFrame({'appID':d2.appID,'positionType':d2.positionType})
    appID_positionType2=appID_positionType2.drop_duplicates()
    appID_positionType=pd.merge(appID_positionType1,appID_positionType2,on=['appID','positionType'],how='outer')
    del appID_positionType1,appID_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_appID_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_positionType_exchange_count']=1
        t1=t1.groupby(['appID','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['appID','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_positionType_all_count']=t3['preday_'+str(p)+'_appID_positionType_not_exchange_count']+t3['preday_'+str(p)+'_appID_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_positionType_exchange_rate']=t3['preday_'+str(p)+'_appID_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_positionType_all_count'].astype('float')
        appID_positionType=pd.merge(appID_positionType,t3,on=['appID','positionType'],how='outer')
    
    del t1,t2,t3
    appID_positionType=appID_positionType.replace(np.nan,0)
    ex=pd.merge(ex,appID_positionType,on=['appID','positionType'],how='left')

    appID_connectionType1=pd.DataFrame({'appID':d1.appID,'connectionType':d1.connectionType})
    appID_connectionType1=appID_connectionType1.drop_duplicates()
    appID_connectionType2=pd.DataFrame({'appID':d2.appID,'connectionType':d2.connectionType})
    appID_connectionType2=appID_connectionType2.drop_duplicates()
    appID_connectionType=pd.merge(appID_connectionType1,appID_connectionType2,on=['appID','connectionType'],how='outer')
    del appID_connectionType1,appID_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_appID_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_connectionType_exchange_count']=1
        t1=t1.groupby(['appID','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['appID','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_connectionType_all_count']=t3['preday_'+str(p)+'_appID_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_appID_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_connectionType_exchange_rate']=t3['preday_'+str(p)+'_appID_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_connectionType_all_count'].astype('float')
        appID_connectionType=pd.merge(appID_connectionType,t3,on=['appID','connectionType'],how='outer')
    
    del t1,t2,t3
    appID_connectionType=appID_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,appID_connectionType,on=['appID','connectionType'],how='left')

    appID_telecomsOperator1=pd.DataFrame({'appID':d1.appID,'telecomsOperator':d1.telecomsOperator})
    appID_telecomsOperator1=appID_telecomsOperator1.drop_duplicates()
    appID_telecomsOperator2=pd.DataFrame({'appID':d2.appID,'telecomsOperator':d2.telecomsOperator})
    appID_telecomsOperator2=appID_telecomsOperator2.drop_duplicates()
    appID_telecomsOperator=pd.merge(appID_telecomsOperator1,appID_telecomsOperator2,on=['appID','telecomsOperator'],how='outer')
    del appID_telecomsOperator1,appID_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_appID_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['appID','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['appID','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_telecomsOperator_all_count']=t3['preday_'+str(p)+'_appID_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_appID_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_appID_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_telecomsOperator_all_count'].astype('float')
        appID_telecomsOperator=pd.merge(appID_telecomsOperator,t3,on=['appID','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    appID_telecomsOperator=appID_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,appID_telecomsOperator,on=['appID','telecomsOperator'],how='left')

    appID_appPlatform1=pd.DataFrame({'appID':d1.appID,'appPlatform':d1.appPlatform})
    appID_appPlatform1=appID_appPlatform1.drop_duplicates()
    appID_appPlatform2=pd.DataFrame({'appID':d2.appID,'appPlatform':d2.appPlatform})
    appID_appPlatform2=appID_appPlatform2.drop_duplicates()
    appID_appPlatform=pd.merge(appID_appPlatform1,appID_appPlatform2,on=['appID','appPlatform'],how='outer')
    del appID_appPlatform1,appID_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appID':d3.appID,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'appID':d4.appID,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_appID_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_appID_appPlatform_exchange_count']=1
        t1=t1.groupby(['appID','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['appID','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appID','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appID_appPlatform_all_count']=t3['preday_'+str(p)+'_appID_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_appID_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appID_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appID_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_appID_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appID_appPlatform_all_count'].astype('float')
        appID_appPlatform=pd.merge(appID_appPlatform,t3,on=['appID','appPlatform'],how='outer')
    
    del t1,t2,t3
    appID_appPlatform=appID_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,appID_appPlatform,on=['appID','appPlatform'],how='left')
    
    '''user_ad特征群/appCategory'''
    appCategory_age1=pd.DataFrame({'appCategory':d1.appCategory,'age':d1.age})
    appCategory_age1=appCategory_age1.drop_duplicates()
    appCategory_age2=pd.DataFrame({'appCategory':d2.appCategory,'age':d2.age})
    appCategory_age2=appCategory_age2.drop_duplicates()
    appCategory_age=pd.merge(appCategory_age1,appCategory_age2,on=['appCategory','age'],how='outer')
    del appCategory_age1,appCategory_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_age_exchange_count']=1
        t1=t1.groupby(['appCategory','age']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_age_all_count']=t3['preday_'+str(p)+'_appCategory_age_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_age_exchange_rate']=t3['preday_'+str(p)+'_appCategory_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_age_all_count'].astype('float')
        appCategory_age=pd.merge(appCategory_age,t3,on=['appCategory','age'],how='outer')
    
    del t1,t2,t3
    appCategory_age=appCategory_age.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_age,on=['appCategory','age'],how='left')

    appCategory_gender1=pd.DataFrame({'appCategory':d1.appCategory,'gender':d1.gender})
    appCategory_gender1=appCategory_gender1.drop_duplicates()
    appCategory_gender2=pd.DataFrame({'appCategory':d2.appCategory,'gender':d2.gender})
    appCategory_gender2=appCategory_gender2.drop_duplicates()
    appCategory_gender=pd.merge(appCategory_gender1,appCategory_gender2,on=['appCategory','gender'],how='outer')
    del appCategory_gender1,appCategory_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_gender_exchange_count']=1
        t1=t1.groupby(['appCategory','gender']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_gender_all_count']=t3['preday_'+str(p)+'_appCategory_gender_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_gender_exchange_rate']=t3['preday_'+str(p)+'_appCategory_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_gender_all_count'].astype('float')
        appCategory_gender=pd.merge(appCategory_gender,t3,on=['appCategory','gender'],how='outer')
    
    del t1,t2,t3
    appCategory_gender=appCategory_gender.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_gender,on=['appCategory','gender'],how='left')

    appCategory_education1=pd.DataFrame({'appCategory':d1.appCategory,'education':d1.education})
    appCategory_education1=appCategory_education1.drop_duplicates()
    appCategory_education2=pd.DataFrame({'appCategory':d2.appCategory,'education':d2.education})
    appCategory_education2=appCategory_education2.drop_duplicates()
    appCategory_education=pd.merge(appCategory_education1,appCategory_education2,on=['appCategory','education'],how='outer')
    del appCategory_education1,appCategory_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_education_exchange_count']=1
        t1=t1.groupby(['appCategory','education']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_education_all_count']=t3['preday_'+str(p)+'_appCategory_education_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_education_exchange_rate']=t3['preday_'+str(p)+'_appCategory_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_education_all_count'].astype('float')
        appCategory_education=pd.merge(appCategory_education,t3,on=['appCategory','education'],how='outer')
    
    del t1,t2,t3
    appCategory_education=appCategory_education.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_education,on=['appCategory','education'],how='left')

    appCategory_haveBaby1=pd.DataFrame({'appCategory':d1.appCategory,'haveBaby':d1.haveBaby})
    appCategory_haveBaby1=appCategory_haveBaby1.drop_duplicates()
    appCategory_haveBaby2=pd.DataFrame({'appCategory':d2.appCategory,'haveBaby':d2.haveBaby})
    appCategory_haveBaby2=appCategory_haveBaby2.drop_duplicates()
    appCategory_haveBaby=pd.merge(appCategory_haveBaby1,appCategory_haveBaby2,on=['appCategory','haveBaby'],how='outer')
    del appCategory_haveBaby1,appCategory_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_haveBaby_exchange_count']=1
        t1=t1.groupby(['appCategory','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_haveBaby_all_count']=t3['preday_'+str(p)+'_appCategory_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_appCategory_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_haveBaby_all_count'].astype('float')
        appCategory_haveBaby=pd.merge(appCategory_haveBaby,t3,on=['appCategory','haveBaby'],how='outer')
    
    del t1,t2,t3
    appCategory_haveBaby=appCategory_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_haveBaby,on=['appCategory','haveBaby'],how='left')

    appCategory_marriageStatus1=pd.DataFrame({'appCategory':d1.appCategory,'marriageStatus':d1.marriageStatus})
    appCategory_marriageStatus1=appCategory_marriageStatus1.drop_duplicates()
    appCategory_marriageStatus2=pd.DataFrame({'appCategory':d2.appCategory,'marriageStatus':d2.marriageStatus})
    appCategory_marriageStatus2=appCategory_marriageStatus2.drop_duplicates()
    appCategory_marriageStatus=pd.merge(appCategory_marriageStatus1,appCategory_marriageStatus2,on=['appCategory','marriageStatus'],how='outer')
    del appCategory_marriageStatus1,appCategory_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_marriageStatus_exchange_count']=1
        t1=t1.groupby(['appCategory','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_marriageStatus_all_count']=t3['preday_'+str(p)+'_appCategory_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_appCategory_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_marriageStatus_all_count'].astype('float')
        appCategory_marriageStatus=pd.merge(appCategory_marriageStatus,t3,on=['appCategory','marriageStatus'],how='outer')
    
    del t1,t2,t3
    appCategory_marriageStatus=appCategory_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_marriageStatus,on=['appCategory','marriageStatus'],how='left')
    
    '''ad_position特征群/appCategory'''
    appCategory_positionID1=pd.DataFrame({'appCategory':d1.appCategory,'positionID':d1.positionID})
    appCategory_positionID1=appCategory_positionID1.drop_duplicates()
    appCategory_positionID2=pd.DataFrame({'appCategory':d2.appCategory,'positionID':d2.positionID})
    appCategory_positionID2=appCategory_positionID2.drop_duplicates()
    appCategory_positionID=pd.merge(appCategory_positionID1,appCategory_positionID2,on=['appCategory','positionID'],how='outer')
    del appCategory_positionID1,appCategory_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_positionID_exchange_count']=1
        t1=t1.groupby(['appCategory','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_positionID_all_count']=t3['preday_'+str(p)+'_appCategory_positionID_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_positionID_exchange_rate']=t3['preday_'+str(p)+'_appCategory_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_positionID_all_count'].astype('float')
        appCategory_positionID=pd.merge(appCategory_positionID,t3,on=['appCategory','positionID'],how='outer')
    
    del t1,t2,t3
    appCategory_positionID=appCategory_positionID.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_positionID,on=['appCategory','positionID'],how='left')

    appCategory_sitesetID1=pd.DataFrame({'appCategory':d1.appCategory,'sitesetID':d1.sitesetID})
    appCategory_sitesetID1=appCategory_sitesetID1.drop_duplicates()
    appCategory_sitesetID2=pd.DataFrame({'appCategory':d2.appCategory,'sitesetID':d2.sitesetID})
    appCategory_sitesetID2=appCategory_sitesetID2.drop_duplicates()
    appCategory_sitesetID=pd.merge(appCategory_sitesetID1,appCategory_sitesetID2,on=['appCategory','sitesetID'],how='outer')
    del appCategory_sitesetID1,appCategory_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_sitesetID_exchange_count']=1
        t1=t1.groupby(['appCategory','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_sitesetID_all_count']=t3['preday_'+str(p)+'_appCategory_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_appCategory_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_sitesetID_all_count'].astype('float')
        appCategory_sitesetID=pd.merge(appCategory_sitesetID,t3,on=['appCategory','sitesetID'],how='outer')
    
    del t1,t2,t3
    appCategory_sitesetID=appCategory_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_sitesetID,on=['appCategory','sitesetID'],how='left')

    appCategory_positionType1=pd.DataFrame({'appCategory':d1.appCategory,'positionType':d1.positionType})
    appCategory_positionType1=appCategory_positionType1.drop_duplicates()
    appCategory_positionType2=pd.DataFrame({'appCategory':d2.appCategory,'positionType':d2.positionType})
    appCategory_positionType2=appCategory_positionType2.drop_duplicates()
    appCategory_positionType=pd.merge(appCategory_positionType1,appCategory_positionType2,on=['appCategory','positionType'],how='outer')
    del appCategory_positionType1,appCategory_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_positionType_exchange_count']=1
        t1=t1.groupby(['appCategory','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_positionType_all_count']=t3['preday_'+str(p)+'_appCategory_positionType_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_positionType_exchange_rate']=t3['preday_'+str(p)+'_appCategory_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_positionType_all_count'].astype('float')
        appCategory_positionType=pd.merge(appCategory_positionType,t3,on=['appCategory','positionType'],how='outer')
    
    del t1,t2,t3
    appCategory_positionType=appCategory_positionType.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_positionType,on=['appCategory','positionType'],how='left')

    appCategory_connectionType1=pd.DataFrame({'appCategory':d1.appCategory,'connectionType':d1.connectionType})
    appCategory_connectionType1=appCategory_connectionType1.drop_duplicates()
    appCategory_connectionType2=pd.DataFrame({'appCategory':d2.appCategory,'connectionType':d2.connectionType})
    appCategory_connectionType2=appCategory_connectionType2.drop_duplicates()
    appCategory_connectionType=pd.merge(appCategory_connectionType1,appCategory_connectionType2,on=['appCategory','connectionType'],how='outer')
    del appCategory_connectionType1,appCategory_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_connectionType_exchange_count']=1
        t1=t1.groupby(['appCategory','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_connectionType_all_count']=t3['preday_'+str(p)+'_appCategory_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_connectionType_exchange_rate']=t3['preday_'+str(p)+'_appCategory_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_connectionType_all_count'].astype('float')
        appCategory_connectionType=pd.merge(appCategory_connectionType,t3,on=['appCategory','connectionType'],how='outer')
    
    del t1,t2,t3
    appCategory_connectionType=appCategory_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_connectionType,on=['appCategory','connectionType'],how='left')

    appCategory_telecomsOperator1=pd.DataFrame({'appCategory':d1.appCategory,'telecomsOperator':d1.telecomsOperator})
    appCategory_telecomsOperator1=appCategory_telecomsOperator1.drop_duplicates()
    appCategory_telecomsOperator2=pd.DataFrame({'appCategory':d2.appCategory,'telecomsOperator':d2.telecomsOperator})
    appCategory_telecomsOperator2=appCategory_telecomsOperator2.drop_duplicates()
    appCategory_telecomsOperator=pd.merge(appCategory_telecomsOperator1,appCategory_telecomsOperator2,on=['appCategory','telecomsOperator'],how='outer')
    del appCategory_telecomsOperator1,appCategory_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['appCategory','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_telecomsOperator_all_count']=t3['preday_'+str(p)+'_appCategory_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_appCategory_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_telecomsOperator_all_count'].astype('float')
        appCategory_telecomsOperator=pd.merge(appCategory_telecomsOperator,t3,on=['appCategory','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    appCategory_telecomsOperator=appCategory_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_telecomsOperator,on=['appCategory','telecomsOperator'],how='left')

    appCategory_appPlatform1=pd.DataFrame({'appCategory':d1.appCategory,'appPlatform':d1.appPlatform})
    appCategory_appPlatform1=appCategory_appPlatform1.drop_duplicates()
    appCategory_appPlatform2=pd.DataFrame({'appCategory':d2.appCategory,'appPlatform':d2.appPlatform})
    appCategory_appPlatform2=appCategory_appPlatform2.drop_duplicates()
    appCategory_appPlatform=pd.merge(appCategory_appPlatform1,appCategory_appPlatform2,on=['appCategory','appPlatform'],how='outer')
    del appCategory_appPlatform1,appCategory_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'appCategory':d3.appCategory,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'appCategory':d4.appCategory,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_appCategory_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_appCategory_appPlatform_exchange_count']=1
        t1=t1.groupby(['appCategory','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['appCategory','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['appCategory','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_appCategory_appPlatform_all_count']=t3['preday_'+str(p)+'_appCategory_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_appCategory_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_appCategory_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_appCategory_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_appCategory_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_appCategory_appPlatform_all_count'].astype('float')
        appCategory_appPlatform=pd.merge(appCategory_appPlatform,t3,on=['appCategory','appPlatform'],how='outer')
    
    del t1,t2,t3
    appCategory_appPlatform=appCategory_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,appCategory_appPlatform,on=['appCategory','appPlatform'],how='left')

    '''user_ad特征群/ad'''
    adID_age1=pd.DataFrame({'adID':d1.adID,'age':d1.age})
    adID_age1=adID_age1.drop_duplicates()
    adID_age2=pd.DataFrame({'adID':d2.adID,'age':d2.age})
    adID_age2=adID_age2.drop_duplicates()
    adID_age=pd.merge(adID_age1,adID_age2,on=['adID','age'],how='outer')
    del adID_age1,adID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_adID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_age_exchange_count']=1
        t1=t1.groupby(['adID','age']).agg('sum').reset_index()
        t2=t2.groupby(['adID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_age_all_count']=t3['preday_'+str(p)+'_adID_age_not_exchange_count']+t3['preday_'+str(p)+'_adID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_age_exchange_rate']=t3['preday_'+str(p)+'_adID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_age_all_count'].astype('float')
        adID_age=pd.merge(adID_age,t3,on=['adID','age'],how='outer')
    
    del t1,t2,t3
    adID_age=adID_age.replace(np.nan,0)
    ex=pd.merge(ex,adID_age,on=['adID','age'],how='left')

    adID_gender1=pd.DataFrame({'adID':d1.adID,'gender':d1.gender})
    adID_gender1=adID_gender1.drop_duplicates()
    adID_gender2=pd.DataFrame({'adID':d2.adID,'gender':d2.gender})
    adID_gender2=adID_gender2.drop_duplicates()
    adID_gender=pd.merge(adID_gender1,adID_gender2,on=['adID','gender'],how='outer')
    del adID_gender1,adID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_adID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_gender_exchange_count']=1
        t1=t1.groupby(['adID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['adID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_gender_all_count']=t3['preday_'+str(p)+'_adID_gender_not_exchange_count']+t3['preday_'+str(p)+'_adID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_gender_exchange_rate']=t3['preday_'+str(p)+'_adID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_gender_all_count'].astype('float')
        adID_gender=pd.merge(adID_gender,t3,on=['adID','gender'],how='outer')
    
    del t1,t2,t3
    adID_gender=adID_gender.replace(np.nan,0)
    ex=pd.merge(ex,adID_gender,on=['adID','gender'],how='left')

    adID_education1=pd.DataFrame({'adID':d1.adID,'education':d1.education})
    adID_education1=adID_education1.drop_duplicates()
    adID_education2=pd.DataFrame({'adID':d2.adID,'education':d2.education})
    adID_education2=adID_education2.drop_duplicates()
    adID_education=pd.merge(adID_education1,adID_education2,on=['adID','education'],how='outer')
    del adID_education1,adID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_adID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_education_exchange_count']=1
        t1=t1.groupby(['adID','education']).agg('sum').reset_index()
        t2=t2.groupby(['adID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_education_all_count']=t3['preday_'+str(p)+'_adID_education_not_exchange_count']+t3['preday_'+str(p)+'_adID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_education_exchange_rate']=t3['preday_'+str(p)+'_adID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_education_all_count'].astype('float')
        adID_education=pd.merge(adID_education,t3,on=['adID','education'],how='outer')
    
    del t1,t2,t3
    adID_education=adID_education.replace(np.nan,0)
    ex=pd.merge(ex,adID_education,on=['adID','education'],how='left')

    adID_haveBaby1=pd.DataFrame({'adID':d1.adID,'haveBaby':d1.haveBaby})
    adID_haveBaby1=adID_haveBaby1.drop_duplicates()
    adID_haveBaby2=pd.DataFrame({'adID':d2.adID,'haveBaby':d2.haveBaby})
    adID_haveBaby2=adID_haveBaby2.drop_duplicates()
    adID_haveBaby=pd.merge(adID_haveBaby1,adID_haveBaby2,on=['adID','haveBaby'],how='outer')
    del adID_haveBaby1,adID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_adID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_haveBaby_exchange_count']=1
        t1=t1.groupby(['adID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['adID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_haveBaby_all_count']=t3['preday_'+str(p)+'_adID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_adID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_adID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_haveBaby_all_count'].astype('float')
        adID_haveBaby=pd.merge(adID_haveBaby,t3,on=['adID','haveBaby'],how='outer')
    
    del t1,t2,t3
    adID_haveBaby=adID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,adID_haveBaby,on=['adID','haveBaby'],how='left')

    adID_marriageStatus1=pd.DataFrame({'adID':d1.adID,'marriageStatus':d1.marriageStatus})
    adID_marriageStatus1=adID_marriageStatus1.drop_duplicates()
    adID_marriageStatus2=pd.DataFrame({'adID':d2.adID,'marriageStatus':d2.marriageStatus})
    adID_marriageStatus2=adID_marriageStatus2.drop_duplicates()
    adID_marriageStatus=pd.merge(adID_marriageStatus1,adID_marriageStatus2,on=['adID','marriageStatus'],how='outer')
    del adID_marriageStatus1,adID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_adID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['adID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['adID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_marriageStatus_all_count']=t3['preday_'+str(p)+'_adID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_adID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_adID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_marriageStatus_all_count'].astype('float')
        adID_marriageStatus=pd.merge(adID_marriageStatus,t3,on=['adID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    adID_marriageStatus=adID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,adID_marriageStatus,on=['adID','marriageStatus'],how='left')
    
    '''ad_position特征群/ad'''
    adID_positionID1=pd.DataFrame({'adID':d1.adID,'positionID':d1.positionID})
    adID_positionID1=adID_positionID1.drop_duplicates()
    adID_positionID2=pd.DataFrame({'adID':d2.adID,'positionID':d2.positionID})
    adID_positionID2=adID_positionID2.drop_duplicates()
    adID_positionID=pd.merge(adID_positionID1,adID_positionID2,on=['adID','positionID'],how='outer')
    del adID_positionID1,adID_positionID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'positionID':d3.positionID,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'positionID':d4.positionID,'day':d4.day})
        t1['preday_'+str(p)+'_adID_positionID_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_positionID_exchange_count']=1
        t1=t1.groupby(['adID','positionID']).agg('sum').reset_index()
        t2=t2.groupby(['adID','positionID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','positionID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_positionID_all_count']=t3['preday_'+str(p)+'_adID_positionID_not_exchange_count']+t3['preday_'+str(p)+'_adID_positionID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_positionID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_positionID_exchange_rate']=t3['preday_'+str(p)+'_adID_positionID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_positionID_all_count'].astype('float')
        adID_positionID=pd.merge(adID_positionID,t3,on=['adID','positionID'],how='outer')
    
    del t1,t2,t3
    adID_positionID=adID_positionID.replace(np.nan,0)
    ex=pd.merge(ex,adID_positionID,on=['adID','positionID'],how='left')

    adID_sitesetID1=pd.DataFrame({'adID':d1.adID,'sitesetID':d1.sitesetID})
    adID_sitesetID1=adID_sitesetID1.drop_duplicates()
    adID_sitesetID2=pd.DataFrame({'adID':d2.adID,'sitesetID':d2.sitesetID})
    adID_sitesetID2=adID_sitesetID2.drop_duplicates()
    adID_sitesetID=pd.merge(adID_sitesetID1,adID_sitesetID2,on=['adID','sitesetID'],how='outer')
    del adID_sitesetID1,adID_sitesetID2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'sitesetID':d3.sitesetID,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'sitesetID':d4.sitesetID,'day':d4.day})
        t1['preday_'+str(p)+'_adID_sitesetID_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_sitesetID_exchange_count']=1
        t1=t1.groupby(['adID','sitesetID']).agg('sum').reset_index()
        t2=t2.groupby(['adID','sitesetID']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','sitesetID'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_sitesetID_all_count']=t3['preday_'+str(p)+'_adID_sitesetID_not_exchange_count']+t3['preday_'+str(p)+'_adID_sitesetID_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_sitesetID_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_sitesetID_exchange_rate']=t3['preday_'+str(p)+'_adID_sitesetID_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_sitesetID_all_count'].astype('float')
        adID_sitesetID=pd.merge(adID_sitesetID,t3,on=['adID','sitesetID'],how='outer')
    
    del t1,t2,t3
    adID_sitesetID=adID_sitesetID.replace(np.nan,0)
    ex=pd.merge(ex,adID_sitesetID,on=['adID','sitesetID'],how='left')

    adID_positionType1=pd.DataFrame({'adID':d1.adID,'positionType':d1.positionType})
    adID_positionType1=adID_positionType1.drop_duplicates()
    adID_positionType2=pd.DataFrame({'adID':d2.adID,'positionType':d2.positionType})
    adID_positionType2=adID_positionType2.drop_duplicates()
    adID_positionType=pd.merge(adID_positionType1,adID_positionType2,on=['adID','positionType'],how='outer')
    del adID_positionType1,adID_positionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'positionType':d3.positionType,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'positionType':d4.positionType,'day':d4.day})
        t1['preday_'+str(p)+'_adID_positionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_positionType_exchange_count']=1
        t1=t1.groupby(['adID','positionType']).agg('sum').reset_index()
        t2=t2.groupby(['adID','positionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','positionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_positionType_all_count']=t3['preday_'+str(p)+'_adID_positionType_not_exchange_count']+t3['preday_'+str(p)+'_adID_positionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_positionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_positionType_exchange_rate']=t3['preday_'+str(p)+'_adID_positionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_positionType_all_count'].astype('float')
        adID_positionType=pd.merge(adID_positionType,t3,on=['adID','positionType'],how='outer')
    
    del t1,t2,t3
    adID_positionType=adID_positionType.replace(np.nan,0)
    ex=pd.merge(ex,adID_positionType,on=['adID','positionType'],how='left')

    adID_connectionType1=pd.DataFrame({'adID':d1.adID,'connectionType':d1.connectionType})
    adID_connectionType1=adID_connectionType1.drop_duplicates()
    adID_connectionType2=pd.DataFrame({'adID':d2.adID,'connectionType':d2.connectionType})
    adID_connectionType2=adID_connectionType2.drop_duplicates()
    adID_connectionType=pd.merge(adID_connectionType1,adID_connectionType2,on=['adID','connectionType'],how='outer')
    del adID_connectionType1,adID_connectionType2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'connectionType':d3.connectionType,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'connectionType':d4.connectionType,'day':d4.day})
        t1['preday_'+str(p)+'_adID_connectionType_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_connectionType_exchange_count']=1
        t1=t1.groupby(['adID','connectionType']).agg('sum').reset_index()
        t2=t2.groupby(['adID','connectionType']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','connectionType'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_connectionType_all_count']=t3['preday_'+str(p)+'_adID_connectionType_not_exchange_count']+t3['preday_'+str(p)+'_adID_connectionType_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_connectionType_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_connectionType_exchange_rate']=t3['preday_'+str(p)+'_adID_connectionType_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_connectionType_all_count'].astype('float')
        adID_connectionType=pd.merge(adID_connectionType,t3,on=['adID','connectionType'],how='outer')
    
    del t1,t2,t3
    adID_connectionType=adID_connectionType.replace(np.nan,0)
    ex=pd.merge(ex,adID_connectionType,on=['adID','connectionType'],how='left')

    adID_telecomsOperator1=pd.DataFrame({'adID':d1.adID,'telecomsOperator':d1.telecomsOperator})
    adID_telecomsOperator1=adID_telecomsOperator1.drop_duplicates()
    adID_telecomsOperator2=pd.DataFrame({'adID':d2.adID,'telecomsOperator':d2.telecomsOperator})
    adID_telecomsOperator2=adID_telecomsOperator2.drop_duplicates()
    adID_telecomsOperator=pd.merge(adID_telecomsOperator1,adID_telecomsOperator2,on=['adID','telecomsOperator'],how='outer')
    del adID_telecomsOperator1,adID_telecomsOperator2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'telecomsOperator':d3.telecomsOperator,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'telecomsOperator':d4.telecomsOperator,'day':d4.day})
        t1['preday_'+str(p)+'_adID_telecomsOperator_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_telecomsOperator_exchange_count']=1
        t1=t1.groupby(['adID','telecomsOperator']).agg('sum').reset_index()
        t2=t2.groupby(['adID','telecomsOperator']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','telecomsOperator'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_telecomsOperator_all_count']=t3['preday_'+str(p)+'_adID_telecomsOperator_not_exchange_count']+t3['preday_'+str(p)+'_adID_telecomsOperator_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_telecomsOperator_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_telecomsOperator_exchange_rate']=t3['preday_'+str(p)+'_adID_telecomsOperator_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_telecomsOperator_all_count'].astype('float')
        adID_telecomsOperator=pd.merge(adID_telecomsOperator,t3,on=['adID','telecomsOperator'],how='outer')
    
    del t1,t2,t3
    adID_telecomsOperator=adID_telecomsOperator.replace(np.nan,0)
    ex=pd.merge(ex,adID_telecomsOperator,on=['adID','telecomsOperator'],how='left')

    adID_appPlatform1=pd.DataFrame({'adID':d1.adID,'appPlatform':d1.appPlatform})
    adID_appPlatform1=adID_appPlatform1.drop_duplicates()
    adID_appPlatform2=pd.DataFrame({'adID':d2.adID,'appPlatform':d2.appPlatform})
    adID_appPlatform2=adID_appPlatform2.drop_duplicates()
    adID_appPlatform=pd.merge(adID_appPlatform1,adID_appPlatform2,on=['adID','appPlatform'],how='outer')
    del adID_appPlatform1,adID_appPlatform2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'adID':d3.adID,'appPlatform':d3.appPlatform,'day':d3.day})
        t2=pd.DataFrame({'adID':d4.adID,'appPlatform':d4.appPlatform,'day':d4.day})
        t1['preday_'+str(p)+'_adID_appPlatform_not_exchange_count']=1
        t2['preday_'+str(p)+'_adID_appPlatform_exchange_count']=1
        t1=t1.groupby(['adID','appPlatform']).agg('sum').reset_index()
        t2=t2.groupby(['adID','appPlatform']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['adID','appPlatform'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_adID_appPlatform_all_count']=t3['preday_'+str(p)+'_adID_appPlatform_not_exchange_count']+t3['preday_'+str(p)+'_adID_appPlatform_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_adID_appPlatform_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_adID_appPlatform_exchange_rate']=t3['preday_'+str(p)+'_adID_appPlatform_exchange_count'].astype('float')/t3['preday_'+str(p)+'_adID_appPlatform_all_count'].astype('float')
        adID_appPlatform=pd.merge(adID_appPlatform,t3,on=['adID','appPlatform'],how='outer')
    
    del t1,t2,t3
    adID_appPlatform=adID_appPlatform.replace(np.nan,0)
    ex=pd.merge(ex,adID_appPlatform,on=['adID','appPlatform'],how='left')

    '''user_position特征群/position'''
    positionID_age1=pd.DataFrame({'positionID':d1.positionID,'age':d1.age})
    positionID_age1=positionID_age1.drop_duplicates()
    positionID_age2=pd.DataFrame({'positionID':d2.positionID,'age':d2.age})
    positionID_age2=positionID_age2.drop_duplicates()
    positionID_age=pd.merge(positionID_age1,positionID_age2,on=['positionID','age'],how='outer')
    del positionID_age1,positionID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionID':d3.positionID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'positionID':d4.positionID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_positionID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionID_age_exchange_count']=1
        t1=t1.groupby(['positionID','age']).agg('sum').reset_index()
        t2=t2.groupby(['positionID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionID_age_all_count']=t3['preday_'+str(p)+'_positionID_age_not_exchange_count']+t3['preday_'+str(p)+'_positionID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionID_age_exchange_rate']=t3['preday_'+str(p)+'_positionID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionID_age_all_count'].astype('float')
        positionID_age=pd.merge(positionID_age,t3,on=['positionID','age'],how='outer')
    
    del t1,t2,t3
    positionID_age=positionID_age.replace(np.nan,0)
    ex=pd.merge(ex,positionID_age,on=['positionID','age'],how='left')

    positionID_gender1=pd.DataFrame({'positionID':d1.positionID,'gender':d1.gender})
    positionID_gender1=positionID_gender1.drop_duplicates()
    positionID_gender2=pd.DataFrame({'positionID':d2.positionID,'gender':d2.gender})
    positionID_gender2=positionID_gender2.drop_duplicates()
    positionID_gender=pd.merge(positionID_gender1,positionID_gender2,on=['positionID','gender'],how='outer')
    del positionID_gender1,positionID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionID':d3.positionID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'positionID':d4.positionID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_positionID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionID_gender_exchange_count']=1
        t1=t1.groupby(['positionID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['positionID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionID_gender_all_count']=t3['preday_'+str(p)+'_positionID_gender_not_exchange_count']+t3['preday_'+str(p)+'_positionID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionID_gender_exchange_rate']=t3['preday_'+str(p)+'_positionID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionID_gender_all_count'].astype('float')
        positionID_gender=pd.merge(positionID_gender,t3,on=['positionID','gender'],how='outer')
    
    del t1,t2,t3
    positionID_gender=positionID_gender.replace(np.nan,0)
    ex=pd.merge(ex,positionID_gender,on=['positionID','gender'],how='left')

    positionID_education1=pd.DataFrame({'positionID':d1.positionID,'education':d1.education})
    positionID_education1=positionID_education1.drop_duplicates()
    positionID_education2=pd.DataFrame({'positionID':d2.positionID,'education':d2.education})
    positionID_education2=positionID_education2.drop_duplicates()
    positionID_education=pd.merge(positionID_education1,positionID_education2,on=['positionID','education'],how='outer')
    del positionID_education1,positionID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionID':d3.positionID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'positionID':d4.positionID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_positionID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionID_education_exchange_count']=1
        t1=t1.groupby(['positionID','education']).agg('sum').reset_index()
        t2=t2.groupby(['positionID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionID_education_all_count']=t3['preday_'+str(p)+'_positionID_education_not_exchange_count']+t3['preday_'+str(p)+'_positionID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionID_education_exchange_rate']=t3['preday_'+str(p)+'_positionID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionID_education_all_count'].astype('float')
        positionID_education=pd.merge(positionID_education,t3,on=['positionID','education'],how='outer')
    
    del t1,t2,t3
    positionID_education=positionID_education.replace(np.nan,0)
    ex=pd.merge(ex,positionID_education,on=['positionID','education'],how='left')

    positionID_haveBaby1=pd.DataFrame({'positionID':d1.positionID,'haveBaby':d1.haveBaby})
    positionID_haveBaby1=positionID_haveBaby1.drop_duplicates()
    positionID_haveBaby2=pd.DataFrame({'positionID':d2.positionID,'haveBaby':d2.haveBaby})
    positionID_haveBaby2=positionID_haveBaby2.drop_duplicates()
    positionID_haveBaby=pd.merge(positionID_haveBaby1,positionID_haveBaby2,on=['positionID','haveBaby'],how='outer')
    del positionID_haveBaby1,positionID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionID':d3.positionID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'positionID':d4.positionID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_positionID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionID_haveBaby_exchange_count']=1
        t1=t1.groupby(['positionID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['positionID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionID_haveBaby_all_count']=t3['preday_'+str(p)+'_positionID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_positionID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_positionID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionID_haveBaby_all_count'].astype('float')
        positionID_haveBaby=pd.merge(positionID_haveBaby,t3,on=['positionID','haveBaby'],how='outer')
    
    del t1,t2,t3
    positionID_haveBaby=positionID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,positionID_haveBaby,on=['positionID','haveBaby'],how='left')

    positionID_marriageStatus1=pd.DataFrame({'positionID':d1.positionID,'marriageStatus':d1.marriageStatus})
    positionID_marriageStatus1=positionID_marriageStatus1.drop_duplicates()
    positionID_marriageStatus2=pd.DataFrame({'positionID':d2.positionID,'marriageStatus':d2.marriageStatus})
    positionID_marriageStatus2=positionID_marriageStatus2.drop_duplicates()
    positionID_marriageStatus=pd.merge(positionID_marriageStatus1,positionID_marriageStatus2,on=['positionID','marriageStatus'],how='outer')
    del positionID_marriageStatus1,positionID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionID':d3.positionID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'positionID':d4.positionID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_positionID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['positionID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['positionID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionID_marriageStatus_all_count']=t3['preday_'+str(p)+'_positionID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_positionID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_positionID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionID_marriageStatus_all_count'].astype('float')
        positionID_marriageStatus=pd.merge(positionID_marriageStatus,t3,on=['positionID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    positionID_marriageStatus=positionID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,positionID_marriageStatus,on=['positionID','marriageStatus'],how='left')

    '''user_position特征群/siteset'''
    sitesetID_age1=pd.DataFrame({'sitesetID':d1.sitesetID,'age':d1.age})
    sitesetID_age1=sitesetID_age1.drop_duplicates()
    sitesetID_age2=pd.DataFrame({'sitesetID':d2.sitesetID,'age':d2.age})
    sitesetID_age2=sitesetID_age2.drop_duplicates()
    sitesetID_age=pd.merge(sitesetID_age1,sitesetID_age2,on=['sitesetID','age'],how='outer')
    del sitesetID_age1,sitesetID_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'sitesetID':d3.sitesetID,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'sitesetID':d4.sitesetID,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_sitesetID_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_sitesetID_age_exchange_count']=1
        t1=t1.groupby(['sitesetID','age']).agg('sum').reset_index()
        t2=t2.groupby(['sitesetID','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['sitesetID','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_sitesetID_age_all_count']=t3['preday_'+str(p)+'_sitesetID_age_not_exchange_count']+t3['preday_'+str(p)+'_sitesetID_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_sitesetID_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_sitesetID_age_exchange_rate']=t3['preday_'+str(p)+'_sitesetID_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_sitesetID_age_all_count'].astype('float')
        sitesetID_age=pd.merge(sitesetID_age,t3,on=['sitesetID','age'],how='outer')
    
    del t1,t2,t3
    sitesetID_age=sitesetID_age.replace(np.nan,0)
    ex=pd.merge(ex,sitesetID_age,on=['sitesetID','age'],how='left')

    sitesetID_gender1=pd.DataFrame({'sitesetID':d1.sitesetID,'gender':d1.gender})
    sitesetID_gender1=sitesetID_gender1.drop_duplicates()
    sitesetID_gender2=pd.DataFrame({'sitesetID':d2.sitesetID,'gender':d2.gender})
    sitesetID_gender2=sitesetID_gender2.drop_duplicates()
    sitesetID_gender=pd.merge(sitesetID_gender1,sitesetID_gender2,on=['sitesetID','gender'],how='outer')
    del sitesetID_gender1,sitesetID_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'sitesetID':d3.sitesetID,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'sitesetID':d4.sitesetID,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_sitesetID_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_sitesetID_gender_exchange_count']=1
        t1=t1.groupby(['sitesetID','gender']).agg('sum').reset_index()
        t2=t2.groupby(['sitesetID','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['sitesetID','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_sitesetID_gender_all_count']=t3['preday_'+str(p)+'_sitesetID_gender_not_exchange_count']+t3['preday_'+str(p)+'_sitesetID_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_sitesetID_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_sitesetID_gender_exchange_rate']=t3['preday_'+str(p)+'_sitesetID_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_sitesetID_gender_all_count'].astype('float')
        sitesetID_gender=pd.merge(sitesetID_gender,t3,on=['sitesetID','gender'],how='outer')
    
    del t1,t2,t3
    sitesetID_gender=sitesetID_gender.replace(np.nan,0)
    ex=pd.merge(ex,sitesetID_gender,on=['sitesetID','gender'],how='left')

    sitesetID_education1=pd.DataFrame({'sitesetID':d1.sitesetID,'education':d1.education})
    sitesetID_education1=sitesetID_education1.drop_duplicates()
    sitesetID_education2=pd.DataFrame({'sitesetID':d2.sitesetID,'education':d2.education})
    sitesetID_education2=sitesetID_education2.drop_duplicates()
    sitesetID_education=pd.merge(sitesetID_education1,sitesetID_education2,on=['sitesetID','education'],how='outer')
    del sitesetID_education1,sitesetID_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'sitesetID':d3.sitesetID,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'sitesetID':d4.sitesetID,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_sitesetID_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_sitesetID_education_exchange_count']=1
        t1=t1.groupby(['sitesetID','education']).agg('sum').reset_index()
        t2=t2.groupby(['sitesetID','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['sitesetID','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_sitesetID_education_all_count']=t3['preday_'+str(p)+'_sitesetID_education_not_exchange_count']+t3['preday_'+str(p)+'_sitesetID_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_sitesetID_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_sitesetID_education_exchange_rate']=t3['preday_'+str(p)+'_sitesetID_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_sitesetID_education_all_count'].astype('float')
        sitesetID_education=pd.merge(sitesetID_education,t3,on=['sitesetID','education'],how='outer')
    
    del t1,t2,t3
    sitesetID_education=sitesetID_education.replace(np.nan,0)
    ex=pd.merge(ex,sitesetID_education,on=['sitesetID','education'],how='left')

    sitesetID_haveBaby1=pd.DataFrame({'sitesetID':d1.sitesetID,'haveBaby':d1.haveBaby})
    sitesetID_haveBaby1=sitesetID_haveBaby1.drop_duplicates()
    sitesetID_haveBaby2=pd.DataFrame({'sitesetID':d2.sitesetID,'haveBaby':d2.haveBaby})
    sitesetID_haveBaby2=sitesetID_haveBaby2.drop_duplicates()
    sitesetID_haveBaby=pd.merge(sitesetID_haveBaby1,sitesetID_haveBaby2,on=['sitesetID','haveBaby'],how='outer')
    del sitesetID_haveBaby1,sitesetID_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'sitesetID':d3.sitesetID,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'sitesetID':d4.sitesetID,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_sitesetID_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_sitesetID_haveBaby_exchange_count']=1
        t1=t1.groupby(['sitesetID','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['sitesetID','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['sitesetID','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_sitesetID_haveBaby_all_count']=t3['preday_'+str(p)+'_sitesetID_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_sitesetID_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_sitesetID_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_sitesetID_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_sitesetID_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_sitesetID_haveBaby_all_count'].astype('float')
        sitesetID_haveBaby=pd.merge(sitesetID_haveBaby,t3,on=['sitesetID','haveBaby'],how='outer')
    
    del t1,t2,t3
    sitesetID_haveBaby=sitesetID_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,sitesetID_haveBaby,on=['sitesetID','haveBaby'],how='left')

    sitesetID_marriageStatus1=pd.DataFrame({'sitesetID':d1.sitesetID,'marriageStatus':d1.marriageStatus})
    sitesetID_marriageStatus1=sitesetID_marriageStatus1.drop_duplicates()
    sitesetID_marriageStatus2=pd.DataFrame({'sitesetID':d2.sitesetID,'marriageStatus':d2.marriageStatus})
    sitesetID_marriageStatus2=sitesetID_marriageStatus2.drop_duplicates()
    sitesetID_marriageStatus=pd.merge(sitesetID_marriageStatus1,sitesetID_marriageStatus2,on=['sitesetID','marriageStatus'],how='outer')
    del sitesetID_marriageStatus1,sitesetID_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'sitesetID':d3.sitesetID,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'sitesetID':d4.sitesetID,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_sitesetID_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_sitesetID_marriageStatus_exchange_count']=1
        t1=t1.groupby(['sitesetID','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['sitesetID','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['sitesetID','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_sitesetID_marriageStatus_all_count']=t3['preday_'+str(p)+'_sitesetID_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_sitesetID_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_sitesetID_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_sitesetID_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_sitesetID_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_sitesetID_marriageStatus_all_count'].astype('float')
        sitesetID_marriageStatus=pd.merge(sitesetID_marriageStatus,t3,on=['sitesetID','marriageStatus'],how='outer')
    
    del t1,t2,t3
    sitesetID_marriageStatus=sitesetID_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,sitesetID_marriageStatus,on=['sitesetID','marriageStatus'],how='left')
    
    '''user_position特征群/positionType'''
    positionType_age1=pd.DataFrame({'positionType':d1.positionType,'age':d1.age})
    positionType_age1=positionType_age1.drop_duplicates()
    positionType_age2=pd.DataFrame({'positionType':d2.positionType,'age':d2.age})
    positionType_age2=positionType_age2.drop_duplicates()
    positionType_age=pd.merge(positionType_age1,positionType_age2,on=['positionType','age'],how='outer')
    del positionType_age1,positionType_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionType':d3.positionType,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'positionType':d4.positionType,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_positionType_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionType_age_exchange_count']=1
        t1=t1.groupby(['positionType','age']).agg('sum').reset_index()
        t2=t2.groupby(['positionType','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionType','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionType_age_all_count']=t3['preday_'+str(p)+'_positionType_age_not_exchange_count']+t3['preday_'+str(p)+'_positionType_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionType_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionType_age_exchange_rate']=t3['preday_'+str(p)+'_positionType_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionType_age_all_count'].astype('float')
        positionType_age=pd.merge(positionType_age,t3,on=['positionType','age'],how='outer')
    
    del t1,t2,t3
    positionType_age=positionType_age.replace(np.nan,0)
    ex=pd.merge(ex,positionType_age,on=['positionType','age'],how='left')

    positionType_gender1=pd.DataFrame({'positionType':d1.positionType,'gender':d1.gender})
    positionType_gender1=positionType_gender1.drop_duplicates()
    positionType_gender2=pd.DataFrame({'positionType':d2.positionType,'gender':d2.gender})
    positionType_gender2=positionType_gender2.drop_duplicates()
    positionType_gender=pd.merge(positionType_gender1,positionType_gender2,on=['positionType','gender'],how='outer')
    del positionType_gender1,positionType_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionType':d3.positionType,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'positionType':d4.positionType,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_positionType_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionType_gender_exchange_count']=1
        t1=t1.groupby(['positionType','gender']).agg('sum').reset_index()
        t2=t2.groupby(['positionType','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionType','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionType_gender_all_count']=t3['preday_'+str(p)+'_positionType_gender_not_exchange_count']+t3['preday_'+str(p)+'_positionType_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionType_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionType_gender_exchange_rate']=t3['preday_'+str(p)+'_positionType_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionType_gender_all_count'].astype('float')
        positionType_gender=pd.merge(positionType_gender,t3,on=['positionType','gender'],how='outer')
    
    del t1,t2,t3
    positionType_gender=positionType_gender.replace(np.nan,0)
    ex=pd.merge(ex,positionType_gender,on=['positionType','gender'],how='left')

    positionType_education1=pd.DataFrame({'positionType':d1.positionType,'education':d1.education})
    positionType_education1=positionType_education1.drop_duplicates()
    positionType_education2=pd.DataFrame({'positionType':d2.positionType,'education':d2.education})
    positionType_education2=positionType_education2.drop_duplicates()
    positionType_education=pd.merge(positionType_education1,positionType_education2,on=['positionType','education'],how='outer')
    del positionType_education1,positionType_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionType':d3.positionType,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'positionType':d4.positionType,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_positionType_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionType_education_exchange_count']=1
        t1=t1.groupby(['positionType','education']).agg('sum').reset_index()
        t2=t2.groupby(['positionType','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionType','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionType_education_all_count']=t3['preday_'+str(p)+'_positionType_education_not_exchange_count']+t3['preday_'+str(p)+'_positionType_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionType_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionType_education_exchange_rate']=t3['preday_'+str(p)+'_positionType_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionType_education_all_count'].astype('float')
        positionType_education=pd.merge(positionType_education,t3,on=['positionType','education'],how='outer')
    
    del t1,t2,t3
    positionType_education=positionType_education.replace(np.nan,0)
    ex=pd.merge(ex,positionType_education,on=['positionType','education'],how='left')

    positionType_haveBaby1=pd.DataFrame({'positionType':d1.positionType,'haveBaby':d1.haveBaby})
    positionType_haveBaby1=positionType_haveBaby1.drop_duplicates()
    positionType_haveBaby2=pd.DataFrame({'positionType':d2.positionType,'haveBaby':d2.haveBaby})
    positionType_haveBaby2=positionType_haveBaby2.drop_duplicates()
    positionType_haveBaby=pd.merge(positionType_haveBaby1,positionType_haveBaby2,on=['positionType','haveBaby'],how='outer')
    del positionType_haveBaby1,positionType_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionType':d3.positionType,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'positionType':d4.positionType,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_positionType_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionType_haveBaby_exchange_count']=1
        t1=t1.groupby(['positionType','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['positionType','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionType','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionType_haveBaby_all_count']=t3['preday_'+str(p)+'_positionType_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_positionType_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionType_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionType_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_positionType_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionType_haveBaby_all_count'].astype('float')
        positionType_haveBaby=pd.merge(positionType_haveBaby,t3,on=['positionType','haveBaby'],how='outer')
    
    del t1,t2,t3
    positionType_haveBaby=positionType_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,positionType_haveBaby,on=['positionType','haveBaby'],how='left')

    positionType_marriageStatus1=pd.DataFrame({'positionType':d1.positionType,'marriageStatus':d1.marriageStatus})
    positionType_marriageStatus1=positionType_marriageStatus1.drop_duplicates()
    positionType_marriageStatus2=pd.DataFrame({'positionType':d2.positionType,'marriageStatus':d2.marriageStatus})
    positionType_marriageStatus2=positionType_marriageStatus2.drop_duplicates()
    positionType_marriageStatus=pd.merge(positionType_marriageStatus1,positionType_marriageStatus2,on=['positionType','marriageStatus'],how='outer')
    del positionType_marriageStatus1,positionType_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'positionType':d3.positionType,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'positionType':d4.positionType,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_positionType_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_positionType_marriageStatus_exchange_count']=1
        t1=t1.groupby(['positionType','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['positionType','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['positionType','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_positionType_marriageStatus_all_count']=t3['preday_'+str(p)+'_positionType_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_positionType_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_positionType_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_positionType_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_positionType_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_positionType_marriageStatus_all_count'].astype('float')
        positionType_marriageStatus=pd.merge(positionType_marriageStatus,t3,on=['positionType','marriageStatus'],how='outer')
    
    del t1,t2,t3
    positionType_marriageStatus=positionType_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,positionType_marriageStatus,on=['positionType','marriageStatus'],how='left')
    
    '''user_position特征群/connectionType'''
    connectionType_age1=pd.DataFrame({'connectionType':d1.connectionType,'age':d1.age})
    connectionType_age1=connectionType_age1.drop_duplicates()
    connectionType_age2=pd.DataFrame({'connectionType':d2.connectionType,'age':d2.age})
    connectionType_age2=connectionType_age2.drop_duplicates()
    connectionType_age=pd.merge(connectionType_age1,connectionType_age2,on=['connectionType','age'],how='outer')
    del connectionType_age1,connectionType_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'connectionType':d3.connectionType,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'connectionType':d4.connectionType,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_connectionType_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_connectionType_age_exchange_count']=1
        t1=t1.groupby(['connectionType','age']).agg('sum').reset_index()
        t2=t2.groupby(['connectionType','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['connectionType','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_connectionType_age_all_count']=t3['preday_'+str(p)+'_connectionType_age_not_exchange_count']+t3['preday_'+str(p)+'_connectionType_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_connectionType_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_connectionType_age_exchange_rate']=t3['preday_'+str(p)+'_connectionType_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_connectionType_age_all_count'].astype('float')
        connectionType_age=pd.merge(connectionType_age,t3,on=['connectionType','age'],how='outer')
    
    del t1,t2,t3
    connectionType_age=connectionType_age.replace(np.nan,0)
    ex=pd.merge(ex,connectionType_age,on=['connectionType','age'],how='left')

    connectionType_gender1=pd.DataFrame({'connectionType':d1.connectionType,'gender':d1.gender})
    connectionType_gender1=connectionType_gender1.drop_duplicates()
    connectionType_gender2=pd.DataFrame({'connectionType':d2.connectionType,'gender':d2.gender})
    connectionType_gender2=connectionType_gender2.drop_duplicates()
    connectionType_gender=pd.merge(connectionType_gender1,connectionType_gender2,on=['connectionType','gender'],how='outer')
    del connectionType_gender1,connectionType_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'connectionType':d3.connectionType,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'connectionType':d4.connectionType,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_connectionType_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_connectionType_gender_exchange_count']=1
        t1=t1.groupby(['connectionType','gender']).agg('sum').reset_index()
        t2=t2.groupby(['connectionType','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['connectionType','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_connectionType_gender_all_count']=t3['preday_'+str(p)+'_connectionType_gender_not_exchange_count']+t3['preday_'+str(p)+'_connectionType_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_connectionType_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_connectionType_gender_exchange_rate']=t3['preday_'+str(p)+'_connectionType_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_connectionType_gender_all_count'].astype('float')
        connectionType_gender=pd.merge(connectionType_gender,t3,on=['connectionType','gender'],how='outer')
    
    del t1,t2,t3
    connectionType_gender=connectionType_gender.replace(np.nan,0)
    ex=pd.merge(ex,connectionType_gender,on=['connectionType','gender'],how='left')

    connectionType_education1=pd.DataFrame({'connectionType':d1.connectionType,'education':d1.education})
    connectionType_education1=connectionType_education1.drop_duplicates()
    connectionType_education2=pd.DataFrame({'connectionType':d2.connectionType,'education':d2.education})
    connectionType_education2=connectionType_education2.drop_duplicates()
    connectionType_education=pd.merge(connectionType_education1,connectionType_education2,on=['connectionType','education'],how='outer')
    del connectionType_education1,connectionType_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'connectionType':d3.connectionType,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'connectionType':d4.connectionType,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_connectionType_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_connectionType_education_exchange_count']=1
        t1=t1.groupby(['connectionType','education']).agg('sum').reset_index()
        t2=t2.groupby(['connectionType','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['connectionType','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_connectionType_education_all_count']=t3['preday_'+str(p)+'_connectionType_education_not_exchange_count']+t3['preday_'+str(p)+'_connectionType_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_connectionType_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_connectionType_education_exchange_rate']=t3['preday_'+str(p)+'_connectionType_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_connectionType_education_all_count'].astype('float')
        connectionType_education=pd.merge(connectionType_education,t3,on=['connectionType','education'],how='outer')
    
    del t1,t2,t3
    connectionType_education=connectionType_education.replace(np.nan,0)
    ex=pd.merge(ex,connectionType_education,on=['connectionType','education'],how='left')

    connectionType_haveBaby1=pd.DataFrame({'connectionType':d1.connectionType,'haveBaby':d1.haveBaby})
    connectionType_haveBaby1=connectionType_haveBaby1.drop_duplicates()
    connectionType_haveBaby2=pd.DataFrame({'connectionType':d2.connectionType,'haveBaby':d2.haveBaby})
    connectionType_haveBaby2=connectionType_haveBaby2.drop_duplicates()
    connectionType_haveBaby=pd.merge(connectionType_haveBaby1,connectionType_haveBaby2,on=['connectionType','haveBaby'],how='outer')
    del connectionType_haveBaby1,connectionType_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'connectionType':d3.connectionType,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'connectionType':d4.connectionType,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_connectionType_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_connectionType_haveBaby_exchange_count']=1
        t1=t1.groupby(['connectionType','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['connectionType','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['connectionType','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_connectionType_haveBaby_all_count']=t3['preday_'+str(p)+'_connectionType_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_connectionType_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_connectionType_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_connectionType_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_connectionType_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_connectionType_haveBaby_all_count'].astype('float')
        connectionType_haveBaby=pd.merge(connectionType_haveBaby,t3,on=['connectionType','haveBaby'],how='outer')
    
    del t1,t2,t3
    connectionType_haveBaby=connectionType_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,connectionType_haveBaby,on=['connectionType','haveBaby'],how='left')

    connectionType_marriageStatus1=pd.DataFrame({'connectionType':d1.connectionType,'marriageStatus':d1.marriageStatus})
    connectionType_marriageStatus1=connectionType_marriageStatus1.drop_duplicates()
    connectionType_marriageStatus2=pd.DataFrame({'connectionType':d2.connectionType,'marriageStatus':d2.marriageStatus})
    connectionType_marriageStatus2=connectionType_marriageStatus2.drop_duplicates()
    connectionType_marriageStatus=pd.merge(connectionType_marriageStatus1,connectionType_marriageStatus2,on=['connectionType','marriageStatus'],how='outer')
    del connectionType_marriageStatus1,connectionType_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'connectionType':d3.connectionType,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'connectionType':d4.connectionType,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_connectionType_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_connectionType_marriageStatus_exchange_count']=1
        t1=t1.groupby(['connectionType','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['connectionType','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['connectionType','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_connectionType_marriageStatus_all_count']=t3['preday_'+str(p)+'_connectionType_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_connectionType_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_connectionType_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_connectionType_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_connectionType_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_connectionType_marriageStatus_all_count'].astype('float')
        connectionType_marriageStatus=pd.merge(connectionType_marriageStatus,t3,on=['connectionType','marriageStatus'],how='outer')
    
    del t1,t2,t3
    connectionType_marriageStatus=connectionType_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,connectionType_marriageStatus,on=['connectionType','marriageStatus'],how='left')
    
    '''user_position特征群/telecomsOperator'''
    telecomsOperator_age1=pd.DataFrame({'telecomsOperator':d1.telecomsOperator,'age':d1.age})
    telecomsOperator_age1=telecomsOperator_age1.drop_duplicates()
    telecomsOperator_age2=pd.DataFrame({'telecomsOperator':d2.telecomsOperator,'age':d2.age})
    telecomsOperator_age2=telecomsOperator_age2.drop_duplicates()
    telecomsOperator_age=pd.merge(telecomsOperator_age1,telecomsOperator_age2,on=['telecomsOperator','age'],how='outer')
    del telecomsOperator_age1,telecomsOperator_age2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'telecomsOperator':d3.telecomsOperator,'age':d3.age,'day':d3.day})
        t2=pd.DataFrame({'telecomsOperator':d4.telecomsOperator,'age':d4.age,'day':d4.day})
        t1['preday_'+str(p)+'_telecomsOperator_age_not_exchange_count']=1
        t2['preday_'+str(p)+'_telecomsOperator_age_exchange_count']=1
        t1=t1.groupby(['telecomsOperator','age']).agg('sum').reset_index()
        t2=t2.groupby(['telecomsOperator','age']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['telecomsOperator','age'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_telecomsOperator_age_all_count']=t3['preday_'+str(p)+'_telecomsOperator_age_not_exchange_count']+t3['preday_'+str(p)+'_telecomsOperator_age_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_telecomsOperator_age_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_telecomsOperator_age_exchange_rate']=t3['preday_'+str(p)+'_telecomsOperator_age_exchange_count'].astype('float')/t3['preday_'+str(p)+'_telecomsOperator_age_all_count'].astype('float')
        telecomsOperator_age=pd.merge(telecomsOperator_age,t3,on=['telecomsOperator','age'],how='outer')
    
    del t1,t2,t3
    telecomsOperator_age=telecomsOperator_age.replace(np.nan,0)
    ex=pd.merge(ex,telecomsOperator_age,on=['telecomsOperator','age'],how='left')

    telecomsOperator_gender1=pd.DataFrame({'telecomsOperator':d1.telecomsOperator,'gender':d1.gender})
    telecomsOperator_gender1=telecomsOperator_gender1.drop_duplicates()
    telecomsOperator_gender2=pd.DataFrame({'telecomsOperator':d2.telecomsOperator,'gender':d2.gender})
    telecomsOperator_gender2=telecomsOperator_gender2.drop_duplicates()
    telecomsOperator_gender=pd.merge(telecomsOperator_gender1,telecomsOperator_gender2,on=['telecomsOperator','gender'],how='outer')
    del telecomsOperator_gender1,telecomsOperator_gender2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'telecomsOperator':d3.telecomsOperator,'gender':d3.gender,'day':d3.day})
        t2=pd.DataFrame({'telecomsOperator':d4.telecomsOperator,'gender':d4.gender,'day':d4.day})
        t1['preday_'+str(p)+'_telecomsOperator_gender_not_exchange_count']=1
        t2['preday_'+str(p)+'_telecomsOperator_gender_exchange_count']=1
        t1=t1.groupby(['telecomsOperator','gender']).agg('sum').reset_index()
        t2=t2.groupby(['telecomsOperator','gender']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['telecomsOperator','gender'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_telecomsOperator_gender_all_count']=t3['preday_'+str(p)+'_telecomsOperator_gender_not_exchange_count']+t3['preday_'+str(p)+'_telecomsOperator_gender_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_telecomsOperator_gender_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_telecomsOperator_gender_exchange_rate']=t3['preday_'+str(p)+'_telecomsOperator_gender_exchange_count'].astype('float')/t3['preday_'+str(p)+'_telecomsOperator_gender_all_count'].astype('float')
        telecomsOperator_gender=pd.merge(telecomsOperator_gender,t3,on=['telecomsOperator','gender'],how='outer')
    
    del t1,t2,t3
    telecomsOperator_gender=telecomsOperator_gender.replace(np.nan,0)
    ex=pd.merge(ex,telecomsOperator_gender,on=['telecomsOperator','gender'],how='left')

    telecomsOperator_education1=pd.DataFrame({'telecomsOperator':d1.telecomsOperator,'education':d1.education})
    telecomsOperator_education1=telecomsOperator_education1.drop_duplicates()
    telecomsOperator_education2=pd.DataFrame({'telecomsOperator':d2.telecomsOperator,'education':d2.education})
    telecomsOperator_education2=telecomsOperator_education2.drop_duplicates()
    telecomsOperator_education=pd.merge(telecomsOperator_education1,telecomsOperator_education2,on=['telecomsOperator','education'],how='outer')
    del telecomsOperator_education1,telecomsOperator_education2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'telecomsOperator':d3.telecomsOperator,'education':d3.education,'day':d3.day})
        t2=pd.DataFrame({'telecomsOperator':d4.telecomsOperator,'education':d4.education,'day':d4.day})
        t1['preday_'+str(p)+'_telecomsOperator_education_not_exchange_count']=1
        t2['preday_'+str(p)+'_telecomsOperator_education_exchange_count']=1
        t1=t1.groupby(['telecomsOperator','education']).agg('sum').reset_index()
        t2=t2.groupby(['telecomsOperator','education']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['telecomsOperator','education'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_telecomsOperator_education_all_count']=t3['preday_'+str(p)+'_telecomsOperator_education_not_exchange_count']+t3['preday_'+str(p)+'_telecomsOperator_education_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_telecomsOperator_education_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_telecomsOperator_education_exchange_rate']=t3['preday_'+str(p)+'_telecomsOperator_education_exchange_count'].astype('float')/t3['preday_'+str(p)+'_telecomsOperator_education_all_count'].astype('float')
        telecomsOperator_education=pd.merge(telecomsOperator_education,t3,on=['telecomsOperator','education'],how='outer')
    
    del t1,t2,t3
    telecomsOperator_education=telecomsOperator_education.replace(np.nan,0)
    ex=pd.merge(ex,telecomsOperator_education,on=['telecomsOperator','education'],how='left')

    telecomsOperator_haveBaby1=pd.DataFrame({'telecomsOperator':d1.telecomsOperator,'haveBaby':d1.haveBaby})
    telecomsOperator_haveBaby1=telecomsOperator_haveBaby1.drop_duplicates()
    telecomsOperator_haveBaby2=pd.DataFrame({'telecomsOperator':d2.telecomsOperator,'haveBaby':d2.haveBaby})
    telecomsOperator_haveBaby2=telecomsOperator_haveBaby2.drop_duplicates()
    telecomsOperator_haveBaby=pd.merge(telecomsOperator_haveBaby1,telecomsOperator_haveBaby2,on=['telecomsOperator','haveBaby'],how='outer')
    del telecomsOperator_haveBaby1,telecomsOperator_haveBaby2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'telecomsOperator':d3.telecomsOperator,'haveBaby':d3.haveBaby,'day':d3.day})
        t2=pd.DataFrame({'telecomsOperator':d4.telecomsOperator,'haveBaby':d4.haveBaby,'day':d4.day})
        t1['preday_'+str(p)+'_telecomsOperator_haveBaby_not_exchange_count']=1
        t2['preday_'+str(p)+'_telecomsOperator_haveBaby_exchange_count']=1
        t1=t1.groupby(['telecomsOperator','haveBaby']).agg('sum').reset_index()
        t2=t2.groupby(['telecomsOperator','haveBaby']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['telecomsOperator','haveBaby'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_telecomsOperator_haveBaby_all_count']=t3['preday_'+str(p)+'_telecomsOperator_haveBaby_not_exchange_count']+t3['preday_'+str(p)+'_telecomsOperator_haveBaby_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_telecomsOperator_haveBaby_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_telecomsOperator_haveBaby_exchange_rate']=t3['preday_'+str(p)+'_telecomsOperator_haveBaby_exchange_count'].astype('float')/t3['preday_'+str(p)+'_telecomsOperator_haveBaby_all_count'].astype('float')
        telecomsOperator_haveBaby=pd.merge(telecomsOperator_haveBaby,t3,on=['telecomsOperator','haveBaby'],how='outer')
    
    del t1,t2,t3
    telecomsOperator_haveBaby=telecomsOperator_haveBaby.replace(np.nan,0)
    ex=pd.merge(ex,telecomsOperator_haveBaby,on=['telecomsOperator','haveBaby'],how='left')

    telecomsOperator_marriageStatus1=pd.DataFrame({'telecomsOperator':d1.telecomsOperator,'marriageStatus':d1.marriageStatus})
    telecomsOperator_marriageStatus1=telecomsOperator_marriageStatus1.drop_duplicates()
    telecomsOperator_marriageStatus2=pd.DataFrame({'telecomsOperator':d2.telecomsOperator,'marriageStatus':d2.marriageStatus})
    telecomsOperator_marriageStatus2=telecomsOperator_marriageStatus2.drop_duplicates()
    telecomsOperator_marriageStatus=pd.merge(telecomsOperator_marriageStatus1,telecomsOperator_marriageStatus2,on=['telecomsOperator','marriageStatus'],how='outer')
    del telecomsOperator_marriageStatus1,telecomsOperator_marriageStatus2

    for p in range(1,8):
        start_days=end_days-p
        d3=d1[((d1.day>=start_days)&(d1.day<end_days))]
        d4=d2[((d2.day>=start_days)&(d2.day<end_days))]
        t1=pd.DataFrame({'telecomsOperator':d3.telecomsOperator,'marriageStatus':d3.marriageStatus,'day':d3.day})
        t2=pd.DataFrame({'telecomsOperator':d4.telecomsOperator,'marriageStatus':d4.marriageStatus,'day':d4.day})
        t1['preday_'+str(p)+'_telecomsOperator_marriageStatus_not_exchange_count']=1
        t2['preday_'+str(p)+'_telecomsOperator_marriageStatus_exchange_count']=1
        t1=t1.groupby(['telecomsOperator','marriageStatus']).agg('sum').reset_index()
        t2=t2.groupby(['telecomsOperator','marriageStatus']).agg('sum').reset_index()
        t1=t1.drop(['day'],axis=1)
        t2=t2.drop(['day'],axis=1)
        t3=pd.merge(t1,t2,on=['telecomsOperator','marriageStatus'],how='outer')
        t3=t3.replace(np.nan,0)
        t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_all_count']=t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_not_exchange_count']+t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_exchange_count']
        t3=t3.drop(['preday_'+str(p)+'_telecomsOperator_marriageStatus_not_exchange_count'],axis=1)
        t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_exchange_rate']=t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_exchange_count'].astype('float')/t3['preday_'+str(p)+'_telecomsOperator_marriageStatus_all_count'].astype('float')
        telecomsOperator_marriageStatus=pd.merge(telecomsOperator_marriageStatus,t3,on=['telecomsOperator','marriageStatus'],how='outer')
    
    del t1,t2,t3
    telecomsOperator_marriageStatus=telecomsOperator_marriageStatus.replace(np.nan,0)
    ex=pd.merge(ex,telecomsOperator_marriageStatus,on=['telecomsOperator','marriageStatus'],how='left')
    
    return ex
ex1=Get_Cross_Feature(27)
print('27done!')
ex2=Get_Cross_Feature(28)
print('28done!')
ex3=Get_Cross_Feature(29)
print('29done!')

dataset12 = pd.concat([ex1,ex2],axis=0)
del ex1,ex2
dataset12 = pd.concat([dataset12,ex3],axis=0)
del ex3

dataset12_y = dataset12[['label']]
dataset12_x = dataset12.drop(['prohome','proresi','clickTime','label','conversionTime','creativeID','userID','positionID','hometown','residence','adID','camgaignID','advertiserID','appID','appCategory'],axis=1)
del dataset12
dataset3=Get_Cross_Feature(31)
dataset3_pred = dataset3[['label']]
d9=pd.DataFrame({'instanceID':dataset3.instanceID})
d9=d9.sort_values('instanceID')
dataset3_x = dataset3.drop(['prohome','proresi','instanceID','clickTime','label','creativeID','userID','positionID','hometown','residence','adID','camgaignID','advertiserID','appID','appCategory'],axis=1)
del dataset3

dataset4=Get_Cross_Feature(30)
dataset4_y = dataset4[['label']]
dataset4_x = dataset4.drop(['prohome','proresi','clickTime','label','conversionTime','creativeID','userID','positionID','hometown','residence','adID','camgaignID','advertiserID','appID','appCategory'],axis=1)
del dataset4

dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)
dataset4 = xgb.DMatrix(dataset4_x,label=dataset4_y)
del dataset12_x,dataset12_y,dataset3_x,dataset4_x,dataset4_y

params={'booster':'gbtree',
	    'objective':'binary:logistic',
        'eval_metric':'logloss',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':7,
	    'lambda':10,
	    'subsample':0.8,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'seed':0
	    }
watchlist = [(dataset12,'train'),(dataset4,'val')]
model = xgb.train(params,dataset12,num_boost_round=500,evals=watchlist)
del dataset12
dataset3_pred['label'] = model.predict(dataset3)
dataset3_pred=pd.DataFrame({'prob':dataset3_pred.label})
submission=pd.concat([d9,dataset3_pred],axis=1)
submission.to_csv('submission.csv',index=None)
print (dataset3_pred.describe())