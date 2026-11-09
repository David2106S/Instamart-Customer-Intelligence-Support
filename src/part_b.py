from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.base import clone
import joblib
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)

def run():
 c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']); o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); p=pd.read_csv(ROOT/'data/clean/products.csv')
 done=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
 rfm=done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
 rfm['recency']=(snap-rfm.last_order).dt.days; X=rfm[['recency','frequency','monetary']]
 scaler=StandardScaler(); Z=scaler.fit_transform(X)
 scores=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z); scores.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(Z,km.labels_)})
 pd.DataFrame(scores).to_csv(OUT/'kmeans_scores.csv',index=False)
 # K=4 is the reference's business-interpretable selection.
 km=KMeans(n_clusters=4,n_init=10,random_state=42).fit(Z); rfm['segment_id']=km.labels_
 prof=rfm.groupby('segment_id').agg(customers=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).sort_values('monetary',ascending=False)
 premium_id=int(prof.index[0]); names={}
 for sid,row in prof.iterrows():
  if sid==premium_id: names[sid]='Premium'
  elif row.frequency>rfm.frequency.mean(): names[sid]='Frequent Loyalists'
  elif row.recency>60: names[sid]='Dormant Customers'
  elif row.recency>rfm.recency.median(): names[sid]='Occasional Shoppers'
  else: names[sid]='Regular Customers'
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
 rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
 pca=PCA(n_components=2).fit(Z); (OUT/'pca_explained_variance.json').write_text(json.dumps(pca.explained_variance_ratio_.tolist()))
 # Lapse cohort and features strictly before cutoff.
 cutoff=pd.Timestamp('2025-09-01'); hist=o[o.order_ts<cutoff].copy(); hist_done=hist[hist.status.eq('completed')].merge(p[['product_id','category']],on='product_id')
 cohort=hist_done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'),fav_category=('category',lambda s:s.mode().iloc[0])).reset_index()
 canc=hist.groupby('customer_id').apply(lambda g:(g.status=='cancelled').sum()/max(1,g.status.isin(['completed','cancelled']).sum())).rename('cancel_rate').reset_index()
 cohort=cohort.merge(canc,on='customer_id').merge(c,on='customer_id')
 cohort['recency']=(cutoff-cohort.last_order).dt.days; cohort['tenure_days']=(cutoff-cohort.signup_date).dt.days
 sep=o[(o.order_ts>=cutoff)&(o.order_ts<pd.Timestamp('2025-10-01'))&o.status.eq('completed')].groupby('customer_id').size()
 cohort['lapsed']=(~cohort.customer_id.isin(sep.index)).astype(int)
 feature_cols=['recency','frequency','monetary','avg_discount','cancel_rate','avg_delivery','fav_category','city','age','gender','tenure_days']
 X=cohort[feature_cols].copy(); y=cohort.lapsed
 assert not X.isna().any().any(), f'Missing features: {X.columns[X.isna().any()].tolist()}'
 numeric=['recency','frequency','monetary','avg_discount','cancel_rate','avg_delivery','age','tenure_days']; categorical=['fav_category','city','gender']
 pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical)])
 Xtr,Xte,ytr,yte,idxtr,idxte=train_test_split(X,y,cohort.index,test_size=.2,stratify=y,random_state=42)
 log=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000))]); gl=GridSearchCV(log,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
 rf=Pipeline([('pre',pre),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); gr=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
 models={'logistic':gl.best_estimator_,'random_forest':gr.best_estimator_}; report={}
 for name,model in models.items():
  pred=model.predict(Xte); report[name]={'best_params':(gl.best_params_ if name=='logistic' else gr.best_params_),'cv_f1':float(gl.best_score_ if name=='logistic' else gr.best_score_),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)}
 selected=models['random_forest'] if report['random_forest']['f1']>=report['logistic']['f1'] else models['logistic']
 if max(report['random_forest']['f1'],report['logistic']['f1'])<.55: print('WARNING: F1 below target; inspect features/model tuning')
 # training-only oversampling experiment on logistic regression
 bestlog=gl.best_estimator_; minority=np.flatnonzero(ytr.to_numpy()==1); majority=np.flatnonzero(ytr.to_numpy()==0); rng=np.random.RandomState(42)
 add=rng.choice(minority,size=max(0,len(majority)-len(minority)),replace=True); ix=np.concatenate([np.arange(len(ytr)),add])
 bal=clone(bestlog).fit(Xtr.iloc[ix],ytr.iloc[ix]); bp=bal.predict(Xte)
 report['balanced_logistic']={'precision':precision_score(yte,bp,zero_division=0),'recall':recall_score(yte,bp,zero_division=0),'f1':f1_score(yte,bp,zero_division=0)}
 # RF feature importance
 rf_best=gr.best_estimator_; names_out=rf_best.named_steps['pre'].get_feature_names_out(); imp=rf_best.named_steps['model'].feature_importances_
 report['rf_top_features']=[{'feature':str(names_out[i]),'importance':float(imp[i])} for i in np.argsort(imp)[-5:][::-1]]
 lr=gl.best_estimator_; lr_names=lr.named_steps['pre'].get_feature_names_out(); lr_coef=lr.named_steps['model'].coef_[0]
 report['logistic_coefficient_signs']=[{'feature':str(lr_names[i]),'coefficient':float(lr_coef[i]),'sign':'positive' if lr_coef[i]>0 else 'negative'} for i in np.argsort(np.abs(lr_coef))[-3:][::-1]]
 joblib.dump(selected,OUT/'lapse_model.joblib'); reloaded=joblib.load(OUT/'lapse_model.joblib'); assert np.array_equal(selected.predict(Xte),reloaded.predict(Xte))
 proba=reloaded.predict_proba(Xte)[:,list(reloaded.classes_).index(1)]
 test_cohort=cohort.loc[idxte].copy(); test_cohort['lapse_probability']=proba; test_cohort['actually_lapsed']=yte.values
 premium_ids=set(rfm.loc[rfm.is_premium,'customer_id']); retention=test_cohort[test_cohort.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10)
 retention[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT/'retention_list.csv',index=False)
 baseline_pred=np.zeros(len(yte),dtype=int)
 report['baseline']={'accuracy':float((baseline_pred==yte.to_numpy()).mean()),'lapsed_f1':f1_score(yte,baseline_pred,zero_division=0)}
 report.update({'rfm_summary':{'customers':int(len(rfm)),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},'cohort_rows':len(cohort),'lapsed':int(y.sum()),'train_rows':len(Xtr),'test_rows':len(Xte),'full_frequency_mean':float(X.frequency.mean()),'train_scaler_frequency_mean':float(selected.named_steps['pre'].named_transformers_['num'].named_steps['scale'].mean_[1]) if hasattr(selected.named_steps['pre'].named_transformers_['num'].named_steps['scale'],'mean_') else None,'segment_profile':prof.reset_index().to_dict('records'),'pca':pca.explained_variance_ratio_.tolist(),'retention_true_lapsed':int(retention.actually_lapsed.sum())})
 (OUT/'part_b_report.json').write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,indent=2,default=str)); return report
if __name__=='__main__': run()
