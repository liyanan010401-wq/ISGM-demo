import argparse,json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from src.core import *
from src.data import generate

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(1);torch.manual_seed(cfg['seed']);start=time.time();path=ROOT/'data/sample.json'
 if not path.exists():generate(path,cfg['data_seed'])
 records=json.loads(path.read_text())['records'];train=[r for r in records if r['split']=='train'];model=Model(cfg['d']);history=[]
 freeze(model,True)
 for module in [model.image,model.txt,model.disease,model.decoder]:freeze(module,False)
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])
 for epoch in range(cfg['stage_A_epochs']):
  opt.zero_grad();features=[model.encode_image(torch.tensor(r['image'])) for r in train];q=torch.stack([a[1] for a in features]);t=torch.stack([model.txt(target_tokens(r)) for r in train]);sim=F.normalize(q,dim=1)@F.normalize(t,dim=1).T/.2
  con=patient_aware_contrastive_loss(sim,train);cls=[];gen=[];classification_audit=[]
  # Unlike the separate full-report contrastive branch, L_dm uses clause columns.
  candidates=[dict(c,patient_id=r['patient_id']) for r in train for c in r['clauses']]
  ct=torch.stack([model.txt(ids(c['text'])) for c in candidates]) if candidates else q.new_zeros((0,model.d))
  clause_sim=F.normalize(q,dim=1)@F.normalize(ct,dim=1).T/.2
  dm,dm_audit=patient_aware_dm_loss(clause_sim,train,candidates)
  for i,r in enumerate(train):
   H,qq,P=features[i];classification,record=masked_assertion_loss(P,r['clauses']);cls.append(classification);classification_audit.append(record);gen.append(model.decoder.teacher(target_tokens(r),H,H.new_ones(len(H)))[0])
  classification=torch.stack(cls).mean() if cls else q.sum()*0
  loss=con+.1*dm+classification+torch.stack(gen).mean();loss.backward();opt.step();history.append({'stage':'A','epoch':epoch,'loss':float(loss.detach()),'dm':dm_audit,'assertion_supervision':classification_audit})
 # Cached candidate graph after freezing both encoders and assertion heads.
 freeze(model,True);graph=SourceGraph(train,model);cached=[]
 for r in train:
  with torch.no_grad():
   H,q,P=model.encode_image(torch.tensor(r['image']));clauses,audit=graph.retrieve(q,P,r['patient_id'])
  cached.append((r,H,q,P,clauses,audit))
 freeze(model.support,False);freeze(model.decoder,False);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'])
 for epoch in range(cfg['stage_B_epochs']):
  losses=[]
  for r,H,q,P,clauses,audit in cached:
   opt.zero_grad();m0,w,p,_,_=model.context(H,q,P,clauses,True);gen,_=model.decoder.teacher(target_tokens(r),m0,w);target=torch.tensor([weak_support_target(c,r['clauses']) for c in clauses]);sup=F.nll_loss(p.clamp_min(1e-7).log(),target) if len(clauses) else gen*0
   loss=gen+sup;loss.backward();opt.step();losses.append(float(loss.detach()))
  history.append({'stage':'B','epoch':epoch,'loss':sum(losses)/len(losses)})
 freeze(model,True)
 for module in [model.correction,model.rq,model.rk,model.rv,model.side,model.assertion]:freeze(module,False)
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr']*.5);keep_count=0;adapter_grad=[]
 for epoch in range(cfg['stage_C_epochs']):
  losses=[]
  for r,H,q,P,clauses,audit in cached:
   with torch.no_grad():
    m0,w,_,_,_=model.context(H,q,P,clauses,False);draft=model.decoder.generate(m0,w,cfg['max_tokens']);draft_features=model.txt.tokens(draft).detach()
   opt.zero_grad();m1=model.correction(m0,w,draft_features);gen,token_loss=model.decoder.teacher(target_tokens(r),m1,w)
   align=1-F.cosine_similarity(m1[:len(H)].mean(0),q,dim=0);rel=model.relation_loss(m1,w,r['clauses']);positions=kept_positions(target_tokens(r),draft,r['clauses']);keep=token_loss[positions].mean() if positions else token_loss.sum()*0;keep_count+=len(positions)
   loss=gen+.1*align+.1*rel+.1*keep;loss.backward();adapter_grad.append(sum(float(p.grad.abs().sum()) for p in model.correction.parameters() if p.grad is not None));opt.step();losses.append(float(loss.detach()))
  history.append({'stage':'C','epoch':epoch,'loss':sum(losses)/len(losses)})
 freeze(model,True);model.eval();before={k:v.clone() for k,v in model.state_dict().items()};calls=graph.calls;outputs=[]
 for r in records:
  if r['split']=='test':
   prediction=infer(model,graph,torch.tensor(r['image']),r['patient_id'],cfg['max_tokens']);prediction['patient_id']=r['patient_id'];prediction['evaluation_reference']=' '.join(c['text'] for c in r['clauses']);outputs.append(prediction)
 immutable=all(torch.equal(before[k],v) for k,v in model.state_dict().items());metrics={'status':'synthetic smoke test; NOT chest radiology result or clinical system','seconds':time.time()-start,'training_history':history,'test_records':len(outputs),'test_retrieval_calls':graph.calls-calls,'parameters_unchanged_during_test':immutable,'stage_C_adapter_gradient_min':min(adapter_grad),'keep_eligible_token_occurrences':keep_count,'candidate_cache_version':graph.version,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'substitutions':['8x8 geometric nonmedical inputs','8-dimensional tiny encoders and one-head GRU decoder','manually generated atomic fact schema instead of RadGraph parser','fixed demo thresholds, no clinical calibration','support feature 5*d+39+5+4+2=90 rather than1330'],'source_training_patient_ids':[r['patient_id'] for r in train]}
 torch.save(model.state_dict(),args.output/'tiny_model.pt');(args.output/'predictions.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8');(args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps({'status':metrics['status'],'seconds':metrics['seconds']}))
if __name__=='__main__':main()
