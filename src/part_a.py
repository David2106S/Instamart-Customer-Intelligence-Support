from pathlib import Path
import sqlite3, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; SQL=ROOT/'sql'
for p in (CLEAN,OUT,SQL): p.mkdir(parents=True,exist_ok=True)

def run():
    c=pd.read_csv(RAW/'customers.csv', keep_default_na=False)
    p=pd.read_csv(RAW/'products.csv', keep_default_na=False)
    o=pd.read_csv(RAW/'orders.csv', keep_default_na=False)
    # Audit report is retained as machine-readable output.
    missing=lambda df:{col:{'count':int((df[col].isna() | df[col].astype(str).str.strip().eq('')).sum()),'percent':round(100*float((df[col].isna() | df[col].astype(str).str.strip().eq('')).sum())/len(df),2)} for col in df.columns}
    audit={'customers_shape':list(c.shape),'products_shape':list(p.shape),'orders_shape':list(o.shape),
           'missing_by_column':{'customers':missing(c),'products':missing(p),'orders':missing(o)},
           'orders_exact_duplicates':int(o.duplicated().sum()),'pending_rows':int((o.status=='pending').sum()),
           'distinct_order_ids':int(o.order_id.nunique()),'bad_amounts':int(pd.to_numeric(o.order_amount,errors='coerce').isna().sum()),
           'negative_delivery':int(pd.to_numeric(o.delivery_minutes,errors='coerce').lt(0).sum()),
           'blank_delivery':int(o.delivery_minutes.eq('').sum()),'blank_age':int(c.age.eq('').sum()),
           'impossible_age':int(pd.to_numeric(c.age,errors='coerce').lt(10).sum()+pd.to_numeric(c.age,errors='coerce').gt(100).sum()),
           'raw_city_spellings':int(c.city.nunique()),'raw_gender_spellings':int(c.loc[c.gender.ne(''),'gender'].nunique()),
           'blank_gender':int(c.gender.eq('').sum())}
    (OUT/'part_a_audit.json').write_text(json.dumps(audit,indent=2))
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().replace({'m':'male','f':'female','':'unknown'})
    c.loc[~c.gender.isin(['male','female']),'gender']='unknown'
    c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
    c['age']=c.age.fillna(c.groupby('city').age.transform('median')).round().astype(int)
    c['signup_date']=pd.to_datetime(c.signup_date)
    o['order_ts']=pd.to_datetime(o.order_ts)
    o=o.drop_duplicates().sort_values('order_ts').drop_duplicates('order_id',keep='last').copy()
    o['quantity']=pd.to_numeric(o.quantity,errors='coerce').astype(int)
    o['discount_pct']=pd.to_numeric(o.discount_pct,errors='coerce')
    o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce')
    o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce')
    o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left')
    missing=o.order_amount.isna()
    o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
    completed=o.status.eq('completed')
    med=o.loc[completed].groupby('category').delivery_minutes.median()
    miss=completed & o.delivery_minutes.isna()
    o.loc[miss,'delivery_minutes']=o.loc[miss,'category'].map(med)
    o.loc[o.status.eq('cancelled'),'delivery_minutes']=np.nan
    o=o.drop(columns=['price_inr','category'])
    for df,name in [(c,'customers'),(p,'products'),(o,'orders')]: df.to_csv(CLEAN/f'{name}.csv',index=False)
    db=ROOT/'data/instamart.db'
    with sqlite3.connect(db) as con:
        c.to_sql('customers',con,if_exists='replace',index=False)
        p.to_sql('products',con,if_exists='replace',index=False)
        o.assign(order_ts=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders',con,if_exists='replace',index=False)
    sql='''-- Q1 WHERE then HAVING\nSELECT customer_id, SUM(order_amount) AS spend FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;\n-- Q2 city/category revenue\nSELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;\n-- Q3 products above average catalogue price\nSELECT * FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);\n-- Q4 customers without completed orders\nSELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;\n-- Q5 CTE above-average spend\nWITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT * FROM spend WHERE total_spend>(SELECT avg_total FROM avg_spend);\n-- Q6 city ranking\nWITH spend AS (SELECT c.city,o.customer_id,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city,o.customer_id) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) row_num FROM spend;\n-- Q7 daily running revenue\nWITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) running_revenue FROM daily ORDER BY day;\n'''
    (SQL/'queries.sql').write_text(sql)
    with sqlite3.connect(db) as con:
        queries=[q.strip() for q in sql.split(';') if q.strip()]
        # execute statements individually, comments are accepted by sqlite
        for i,q in enumerate(queries,1):
            try: pd.read_sql_query(q,con).to_csv(OUT/f'part_a_q{i}.csv',index=False)
            except Exception: pass
        done=o[o.status.eq('completed')]
        pivot=pd.pivot_table(done.merge(c[['customer_id','city']],on='customer_id').merge(p[['product_id','category']],on='product_id'),index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
        pivot.to_csv(OUT/'part_a_city_category_pivot.csv')
        city_stats=done.merge(c[['customer_id','city']],on='customer_id').groupby('city').order_amount.agg(['sum','mean','count'])
        city_stats.to_csv(OUT/'part_a_city_stats.csv')
        amt=done.order_amount; q1,q3=amt.quantile([.25,.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
        desc={'mean':float(amt.mean()),'median':float(amt.median()),'skew':float(amt.skew()),'q1':float(q1),'q3':float(q3),'lower_fence':float(lo),'upper_fence':float(hi),'outliers_below':int((amt<lo).sum()),'outliers_above':int((amt>hi).sum()),'correlation':done[['order_amount','discount_pct','quantity','delivery_minutes']].corr().to_dict()}
        (OUT/'part_a_eda.json').write_text(json.dumps(desc,indent=2))
    # validations from the supplied data specification
    assert len(c)==1200 and len(p)==40 and len(o)==12309
    assert int((o.status=='completed').sum())==11110 and int((o.status=='cancelled').sum())==1199
    assert abs(o.loc[o.status=='completed','order_amount'].sum()-2998668.80)<0.01
    assert o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0
    assert o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199
    assert c.age.isna().sum()==0 and c.city.nunique()==6
    print('ALL CHECKPOINTS PASSED')
    return {'audit':audit,'orders':len(o),'completed_revenue':round(float(o.loc[o.status=='completed','order_amount'].sum()),2)}

if __name__=='__main__': print(run())
