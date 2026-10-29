"""Real-model HTTP smoke checks. Run only when the Web UI is idle."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE='http://127.0.0.1:7860'
ROOT=Path(__file__).resolve().parents[1]

def api(route,body=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    request=urllib.request.Request(BASE+route,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urllib.request.urlopen(request,timeout=30) as r:return json.load(r)

def done(job):
    for _ in range(240):
        s=api('/api/jobs/'+job)
        if s['status'] in ('done','error','cancelled'):return s
        time.sleep(.25)
    raise AssertionError('job timeout')

req={'model':'qwen3.5-4b','messages':[{'role':'user','content':'只回答数字：1+1等于多少？'}],
     'groups':[{'layer':16,'neurons':'0-3','scale':1},{'layer':17,'neurons':'10-15','scale':1}],
     'compare':True,'max_tokens':16,'temperature':0}
report={}
assert not api('/api/models')['active_job'], '请先停止界面里的实验'
j=api('/api/chat',req)['id'];result=done(j)
assert result['status']=='done',result
assert result['results']['baseline']['text']==result['results']['intervened']['text'],result
report['identity_comparison']=result
print('identity comparison passed',flush=True)
bad={**req,'groups':[{'layer':16,'neurons':'0','scale':0},{'layer':16,'neurons':'0','scale':2}]}
try:api('/api/chat',bad);raise AssertionError('duplicate not rejected')
except urllib.error.HTTPError as e:assert e.code==400
try:api('/api/chat',req,origin='https://example.org');raise AssertionError('origin not rejected')
except urllib.error.HTTPError as e:assert e.code==403
long={**req,'max_tokens':1024,'compare':False,'messages':[{'role':'user','content':'请详细介绍从1到100的每个整数的性质，不要省略。'}],
      'groups':[{'layer':16,'neurons':'0-15','scale':0},{'layer':17,'neurons':'20-31','scale':2}]}
j=api('/api/chat',long)['id']
try:api('/api/chat',req);raise AssertionError('concurrent job not rejected')
except urllib.error.HTTPError as e:assert e.code==409
# Wait until generation has entered the intervention context before cancelling.
for _ in range(160):
    s=api('/api/jobs/'+j)
    if s['results'].get('intervened',{}).get('tokens',0)>=1:break
    if s['status'] in ('error','done'):raise AssertionError(s)
    time.sleep(.1)
api('/api/jobs/'+j+'/cancel',{})
result=done(j);assert result['status']=='cancelled',result
report['cancel_during_intervention']=result
print('cancel and busy checks passed',flush=True)
req['groups']=[];req['compare']=False
result=done(api('/api/chat',req)['id']);assert result['status']=='done',result
assert result['results']['intervened']['text']==report['identity_comparison']['results']['baseline']['text']
report['after_cancel_baseline']=result
req['adapter']='smoke'
result=done(api('/api/chat',req)['id']);assert result['status']=='done',result
report['adapter_selection']=result
req['adapter']='';req['context_limit']=128;req['max_tokens']=128
result=done(api('/api/chat',req)['id']);assert result['status']=='error' and '超过上下文预算' in result['error'],result
report['context_overflow']=result
(ROOT/'outputs/qwen3.5-4b/web_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('adapter switch, restoration and context checks passed',flush=True)
