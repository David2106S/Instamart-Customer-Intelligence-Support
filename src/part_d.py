from pathlib import Path
import sys,re,json,time,uuid
from datetime import datetime,timezone
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; STATE=ROOT/'state'; POL=ROOT/'data/policies'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True); sys.path.insert(0,str(ROOT)); import mock_llm

class PolicyRAG:
 def __init__(self,overlap=10):
  self.chunks=[]
  for path in sorted(POL.glob('*.txt')):
   words=path.read_text(encoding='utf-8').split(); start=0
   while start<len(words):
    chunk=words[start:start+40]
    if not chunk: break
    self.chunks.append({'source':path.name,'text':' '.join(chunk)})
    if start+40>=len(words): break
    start+=40-overlap
  self.vectorizer=TfidfVectorizer(); self.matrix=self.vectorizer.fit_transform([x['text'] for x in self.chunks])
 def retrieve(self,query,k=3):
  q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix).ravel(); ix=np.argsort(scores)[::-1][:k]
  return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in ix]
 def answer(self,question):
  found=self.retrieve(question,3); context='\n'.join(f"[{x['source']}] {x['text']}" for x in found)
  messages=[{'role':'system','content':'POLICY ASSISTANT. Answer only using the supplied policy context. If the answer is not present, say: I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
  response=mock_llm.chat(messages)
  return {'answer':response.choices[0].message.content,'sources':[x['source'] for x in found]}

def tool_get_order(order_id):
 import sqlite3
 with sqlite3.connect(ROOT/'data/instamart.db') as con:
  row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
 if not row:return {'error':'order not found'}
 return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])}
RAG=PolicyRAG(10)
def tool_search_policy(query): return {'results':RAG.retrieve(query,3)}
def tool_calculate_refund(order_amount,delivery_minutes):
 amt=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
 return {'refund_inr':round(amt,2)}
def _ledger(name,order_id,refund_inr):
 row={'timestamp':datetime.now(timezone.utc).isoformat(),'tool_name':name,'order_id':order_id,'refund_inr':float(refund_inr)}
 with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 return {'refund_inr':float(refund_inr)}
def tool_issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def tool_escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOLS_SPEC=[
 {'name':'get_order','description':'Fetch order status, amount and delivery time.','parameters':{'order_id':'string'}},
 {'name':'search_policy','description':'Retrieve relevant refund policy chunks.','parameters':{'query':'string'}},
 {'name':'calculate_refund','description':'Calculate refund using completion and late-delivery rules.','parameters':{'order_amount':'number','delivery_minutes':'number'}},
 {'name':'issue_refund','description':'Record an eligible automatic refund.','parameters':{'order_id':'string','refund_inr':'number'}},
 {'name':'escalate_to_human','description':'Record a refund requiring human approval.','parameters':{'order_id':'string','refund_inr':'number'}}]
TOOL_MAP={'get_order':tool_get_order,'search_policy':tool_search_policy,'calculate_refund':tool_calculate_refund,'issue_refund':tool_issue_refund,'escalate_to_human':tool_escalate_to_human}

def _save(run_id,scenario_id,messages,tokens,step,executed,user_message):
 payload={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':step,'tools_executed':executed,'user_message':user_message}
 (STATE/f'{run_id}.json').write_text(json.dumps(payload,indent=2,default=str))
def _fallback(status,steps,tokens,calls,why):
 return {'status':status,'final_answer':f'{status}: {why} Please hand this case to a human support agent.','steps':steps,'llm_calls':calls,'tokens_used':tokens}

def _continue(payload,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tools=None):
 run_id=payload['run_id']; sid=payload['scenario_id']; messages=payload['messages']; tokens=payload['tokens_used']; steps=payload.get('steps',[]); executed=payload['tools_executed']; calls=payload.get('llm_calls',0); started=time.monotonic(); newly_executed=0
 while calls<max_iterations:
  if time.monotonic()-started>=time_limit_seconds: return _fallback('FALLBACK_TIMEOUT',steps,tokens,calls,'Time limit reached.')
  response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
  msg=response.choices[0].message; tool_calls=getattr(msg,'tool_calls',[]) or []
  if tokens>token_limit: return _fallback('FALLBACK_TOKEN_LIMIT',steps,tokens,calls,'Token limit reached before the requested tool was executed.')
  if not tool_calls:
   answer=msg.content or 'No answer returned.'; status=answer.split(':',1)[0].strip()
   return {'status':status,'final_answer':answer,'steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id}
  assistant={'role':'assistant','content':None,'tool_calls':[]}
  for tc in tool_calls:
   assistant['tool_calls'].append({'id':tc.id,'name':tc.name,'arguments':tc.arguments})
  messages.append(assistant)
  for tc in tool_calls:
   args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
   if tc.name not in TOOL_MAP: observation={'error':'unknown tool'}
   else: observation=TOOL_MAP[tc.name](**args)
   executed.append(tc.name); newly_executed+=1
   steps.append({'tool':tc.name,'arguments':args,'observation':observation})
   messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(observation)})
   _save(run_id,sid,messages,tokens,len(steps),executed,payload['user_message'])
   # Include progress fields in checkpoint so a resumed run does not repeat tools.
   cp=json.loads((STATE/f'{run_id}.json').read_text()); cp.update({'steps':steps,'llm_calls':calls}); (STATE/f'{run_id}.json').write_text(json.dumps(cp,indent=2,default=str))
   if crash_after_tools is not None and newly_executed>=crash_after_tools: raise RuntimeError('SIMULATED_CRASH_AFTER_CHECKPOINT')
  # next loop uses appended tool observations
 return _fallback('FALLBACK_MAX_ITERATIONS',steps,tokens,calls,'Maximum model-call iterations reached.')

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
 run_id=run_id or f'{scenario_id}_{uuid.uuid4().hex[:8]}'
 payload={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT. Follow refund policy. Use tools to verify orders and never issue refunds outside policy.'},{'role':'user','content':user_message}],'tokens_used':0,'tools_executed':[],'steps':[],'llm_calls':0,'user_message':user_message}
 _save(run_id,scenario_id,payload['messages'],0,0,[],user_message)
 return _continue(payload,max_iterations,token_limit,time_limit_seconds,crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
 payload=json.loads((STATE/f'{run_id}.json').read_text()); return _continue(payload,max_iterations,token_limit,time_limit_seconds)

def run():
 # Retrieval evaluation with overlap=10 and overlap=0.
 eval_df=pd.read_csv(ROOT/'data/raw/policy_eval.csv'); rows=[]
 for overlap in (10,0):
  rag=PolicyRAG(overlap)
  for _,r in eval_df.iterrows():
   got=rag.retrieve(r.question,3); ans=rag.answer(r.question)
   rows.append({'overlap':overlap,'question':r.question,'expected_doc_hit':r.expected_doc=='NONE' or r.expected_doc in [x['source'] for x in got],
    'expected_fact_hit':(r.expected_fact in ans['answer']) if r.expected_doc!='NONE' else None,
    'fallback':(ans['answer'].startswith('I could not find') if r.expected_doc=='NONE' else None),'top_sources':[x['source'] for x in got]})
 pd.DataFrame(rows).to_json(OUT/'policy_eval_results.json',orient='records',indent=2)
 scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv',keep_default_na=False); (OUT/'refund_ledger.jsonl').write_text(''); (OUT/'agent_runs.jsonl').write_text('')
 results=[]
 for scenario_index,r in scenarios.iterrows():
  result=run_agent(r.message,r.scenario_id)
  result['expected_outcome']=r.expected_outcome; result['expected_refund_inr']=float(r.expected_refund_inr)
  with (OUT/'agent_runs.jsonl').open('a') as f:f.write(json.dumps({'scenario_id':r.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']})+'\n')
  result['scenario_id']=r.scenario_id
  results.append(result)
 # Safety demonstrations
 demo_token=run_agent(scenarios.iloc[0].message,'S1_TOKEN_LIMIT',token_limit=200)
 demo_time=run_agent(scenarios.iloc[0].message,'S1_TIMEOUT',time_limit_seconds=0)
 # Crash after second tool; resume from saved checkpoint.
 s1=scenarios.iloc[0]; crash_id='S1_CRASH_RESUME'
 try: run_agent(s1.message,'S1',run_id=crash_id,crash_after_tools=2)
 except RuntimeError as e:
  if str(e)!='SIMULATED_CRASH_AFTER_CHECKPOINT': raise
 resumed=resume_agent(crash_id)
 report={'scenarios':[{'scenario_id':r['scenario_id'], 'status':r['status'],'final_answer':r['final_answer'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'tools':[x['tool'] for x in r['steps']]} for r in results],
         'token_demo':demo_token['status'],'timeout_demo':demo_time['status'],'crash_resume':{'status':resumed['status'],'answer':resumed['final_answer'],'tools':resumed['steps']}}
 (OUT/'part_d_report.json').write_text(json.dumps(report,indent=2,default=str)); print(json.dumps(report,indent=2,default=str)); return report
if __name__=='__main__': run()
