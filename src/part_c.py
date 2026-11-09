from pathlib import Path
import sys,re,json,time,random
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT)); import mock_llm
ALLOWED={'billing','delivery','technical','fraud'}
EMAIL=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE=re.compile(r'(?<!\d)(?:\+?91[- ]?)?[6-9]\d{9}(?!\d)')
ORDER=re.compile(r'\bORD-\d{6}\b')

def redact(text):
 counts={}
 for label,pat in [('email',EMAIL),('phone',PHONE),('order_id',ORDER)]:
  text,n=pat.subn({'email':'[EMAIL]','phone':'[PHONE]','order_id':'[ORDER_ID]'}[label],text); counts[label]=n
 return text,counts

def make_system(few):
 base='CLASSIFY support tickets. Classify into exactly one category: billing, delivery, technical, fraud. Return JSON only with keys category and reason.'
 return base+('\nEXAMPLES:\nTicket: unauthorized transaction and hacked wallet. JSON: {"category":"fraud","reason":"unauthorised use"}\nTicket: app crashes and OTP error. JSON: {"category":"technical","reason":"app issue"}' if few else '')

def call_retry(messages,ticket_id,retry_log):
 for attempt in range(4):
  try: return mock_llm.chat(messages)
  except mock_llm.TransientLLMError:
   if attempt==3: raise
   wait=.05*(2**attempt)*random.uniform(.9,1.15); retry_log.append({'ticket_id':ticket_id,'retry':attempt+1,'wait_seconds':wait}); time.sleep(wait)

def classify(text,few,ticket_id,retry_log):
 clean,_=redact(text); assert not EMAIL.search(clean) and not PHONE.search(clean) and not ORDER.search(clean)
 resp=call_retry([{'role':'system','content':make_system(few)},{'role':'user','content':clean}],ticket_id,retry_log)
 raw=resp.choices[0].message.content or ''; used=resp.usage.total_tokens
 try:
  obj=json.loads(raw); cat=obj.get('category')
  if cat in ALLOWED: return cat,'classified_json',used
  return None,'flagged_invalid_output',used
 except Exception:
  found=set(re.findall(r'[a-z]+',raw.lower())) & ALLOWED
  if len(found)==1: return next(iter(found)),'classified_keyword',used
  return None,'flagged_invalid_output',used

def run_one(few,budget=100000):
 t=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False); seg=pd.read_csv(OUT/'customer_segments.csv')
 mock_llm.reset(); random.seed(42); retries=[]; rows=[]; tokens=0
 for _,r in t.iterrows():
  txt=r.text; status=''; cat=None; used=0
  if not txt.strip(): status='skipped_empty'
  elif len(txt.split())*1.3>400: status='flagged_long'
  elif tokens>=budget: status='budget_exhausted'
  else:
   cat,status,used=classify(txt,few,r.ticket_id,retries); tokens+=used
  rows.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':status,'category':cat,'true_category':r.true_category,'tokens_used':used})
 df=pd.DataFrame(rows); classified=df[df.category.notna()]
 accuracy=float((classified.category==classified.true_category).mean()) if len(classified) else 0
 return df,{'classified':len(classified),'accuracy':accuracy,'tokens':tokens,'tokens_per_classified':tokens/max(1,len(classified)),'retries':retries}

def run():
 z,zr=run_one(False); f,fr=run_one(True); z.to_csv(OUT/'triage_zero_shot.csv',index=False); f.to_csv(OUT/'triage_few_shot.csv',index=False)
 budget,_=run_one(False,5000); budget.to_csv(OUT/'triage_budget_demo.csv',index=False)
 # Final routing uses few-shot classification, with large budget.
 seg=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
 final=f.copy(); final['segment_name']=final.customer_id.map(seg.segment_name); final['priority']='NORMAL'; final['route']='support_queue'
 for i,r in final.iterrows():
  if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: final.loc[i,'route']='human_review'
  elif r.category=='fraud': final.loc[i,'priority']='HIGH'; final.loc[i,'route']='security_team'
  elif r.segment_name=='Premium': final.loc[i,'priority']='HIGH'; final.loc[i,'route']='premium_desk'
 final[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(OUT/'triage_results.csv',index=False)
 confusion=pd.crosstab(z.category,z.true_category).to_dict()
 redaction_totals={'email':0,'phone':0,'order_id':0}
 raw_tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
 for raw_text in raw_tickets.text:
  _,cc=redact(raw_text)
  for key,value in cc.items(): redaction_totals[key]+=value
 report={'zero_shot':zr,'few_shot':fr,'redaction_substitutions':redaction_totals,'budget_exhausted_count':int((budget.status=='budget_exhausted').sum()),'zero_shot_confusion':confusion,'few_shot_status_counts':f.status.value_counts().to_dict(),'final_rows':len(final)}
 (OUT/'part_c_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return report
if __name__=='__main__': run()
