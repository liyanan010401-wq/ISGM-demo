"""Tiny source-preserving graph memory; synthetic facts only, no clinical use."""
import copy,hashlib,json,math
import torch
from torch import nn
from torch.nn import functional as F

VOCAB=['<pad>','<bos>','<eos>','<source>','<S>','<C>','<U>','left','right','bilateral','midline','unknown','present','absent','uncertain','opacity','effusion','nodule','lung','.','and']
TOK={t:i for i,t in enumerate(VOCAB)};SIDES=['left','right','bilateral','midline','unknown'];ASSERT=['present','absent','uncertain','unmentioned'];FINDINGS=['opacity','effusion','nodule']
def ids(text):return [TOK.get(t,TOK['unknown']) for t in text.split()]
def text(tokens):return ' '.join(VOCAB[int(i)] for i in tokens if int(i) not in (TOK['<bos>'],TOK['<pad>']))
def freeze(module,flag=True):
 for p in module.parameters():p.requires_grad_(not flag)

def hard_gate(prob,atomic,tau_s=.55,tau_c=.45,rho=.15):
 gs,gc,gu=prob.unbind(-1)
 conflict=gc>=tau_c;supported=(gs>=tau_s)&(gc<tau_c)&atomic
 w=torch.where(conflict,torch.zeros_like(gs),torch.where(supported,gs,rho*gs))
 state=torch.where(conflict,torch.ones_like(gs,dtype=torch.long),torch.where(supported,torch.zeros_like(gs,dtype=torch.long),torch.full_like(gs,2,dtype=torch.long)))
 return w,state

def projected_attention(query,memory,weight,q,k,v):
 d=query.shape[-1];valid=weight>0
 if not bool(valid.any()):return torch.zeros_like(query)
 qq=q(F.layer_norm(query,(d,)));kk=k(F.layer_norm(memory,(d,)));vv=v(F.layer_norm(memory,(d,)))
 return (qq@kk[valid].T/d**.5).softmax(-1)@(vv[valid]*weight[valid,None])

class TextEncoder(nn.Module):
 def __init__(self,d):super().__init__();self.embed=nn.Embedding(len(VOCAB),d);self.proj=nn.Linear(d,d)
 def tokens(self,seq):return self.proj(self.embed(torch.tensor(seq,dtype=torch.long)))
 def forward(self,seq):return self.tokens(seq).mean(0)
class TinyDecoder(nn.Module):
 def __init__(self,d):
  super().__init__();self.d=d;self.embed=nn.Embedding(len(VOCAB),d);self.cell=nn.GRUCell(d,d);self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.head=nn.Linear(2*d,len(VOCAB))
 def step(self,token,h,memory,weight):
  h=self.cell(self.embed(torch.tensor([token])),h[None])[0]
  ctx=projected_attention(h[None],memory,weight,self.q,self.k,self.v)[0]
  return self.head(torch.cat([h,ctx])),h
 def teacher(self,target,memory,weight):
  # Reference enters as previous-token causal prefix, never a whole-report memory.
  h=memory.new_zeros(self.d);previous=TOK['<bos>'];logits=[]
  for token in target:
   out,h=self.step(previous,h,memory,weight);logits.append(out);previous=token
  logits=torch.stack(logits);losses=F.cross_entropy(logits,torch.tensor(target),reduction='none');return losses.mean(),losses
 def generate(self,memory,weight,max_tokens=14):
  with torch.no_grad():
   h=memory.new_zeros(self.d);previous=TOK['<bos>'];output=[]
   for _ in range(max_tokens):
    out,h=self.step(previous,h,memory,weight);previous=int(out.argmax());output.append(previous)
    if previous==TOK['<eos>']:break
  return output
class Correction(nn.Module):
 def __init__(self,d):
  super().__init__();self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.ff=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,d))
 def forward(self,m0,weight,draft_tokens):
  kv=torch.cat([m0,draft_tokens]);weights=torch.cat([weight,weight.new_ones(len(draft_tokens))]);read=projected_attention(m0,kv,weights,self.q,self.k,self.v)
  return m0+self.ff(read) # original key masks/weights reused by every downstream head
class Model(nn.Module):
 def __init__(self,d=8):
  super().__init__();self.d=d;self.image=nn.Sequential(nn.Conv2d(1,d,3,padding=1),nn.Tanh(),nn.AdaptiveAvgPool2d((4,4)));self.txt=TextEncoder(d);self.disease=nn.Linear(d,39)
  self.support=nn.Sequential(nn.Linear(5*d+39+5+4+2,16),nn.GELU(),nn.Linear(16,3));self.decoder=TinyDecoder(d);self.correction=Correction(d)
  self.rq=nn.Linear(d,d);self.rk=nn.Linear(d,d);self.rv=nn.Linear(d,d);self.side=nn.Linear(d,5);self.assertion=nn.Linear(d,3)
 def encode_image(self,image):
  H=self.image(image[None,None])[0].flatten(1).T;q=H.mean(0);P=self.disease(q).reshape(13,3).softmax(-1);return H,q,P
 def support_prob(self,H,q,P,clause):
  t=self.txt(ids(clause['text']));a=(H@t/self.d**.5).softmax(0);local=H.T@a
  side=F.one_hot(torch.tensor(clause['side']),5).float();assertion=F.one_hot(torch.tensor(clause['assertion']),4).float()
  z=torch.cat([q,t,local,(q-t).abs(),q*t,P.flatten(),side,assertion,q.new_tensor([1.,1.])])
  return self.support(z).softmax(-1)
 def context(self,H,q,P,clauses,soft=False):
  memory=[H];weights=[H.new_ones(len(H))];probs=[];states=[];lengths=[]
  for c in clauses:
   p=self.support_prob(H,q,P,c);probs.append(p);w,state=hard_gate(p,torch.tensor(c['atomic']))
   if soft:w=(1-p[1])*(p[0]+.15*p[2])
   token=[TOK['<source>'],TOK[['<S>','<C>','<U>'][int(state)]]]+ids(c['text']);encoded=self.txt.tokens(token)
   memory.append(encoded);weights.append(w.expand(len(token)));states.append(int(state));lengths.append(len(token))
  return torch.cat(memory),torch.cat(weights),torch.stack(probs) if probs else H.new_zeros(0,3),states,lengths
 def relation_loss(self,m1,weight,relations):
  losses=[]
  groups={}
  for r in relations:groups.setdefault((r['finding'],r['location']),[]).append(r)
  for (finding,location),group in groups.items():
   query=self.txt(ids(FINDINGS[finding]+' '+location)).detach()[None]
   h=projected_attention(query,m1,weight,self.rq,self.rk,self.rv)[0]
   for field,head in [('side',self.side),('assertion',self.assertion)]:
    values={r[field] for r in group}
    if len(values)==1 and not(field=='assertion' and next(iter(values))==3):losses.append(F.cross_entropy(head(h)[None],torch.tensor([next(iter(values))])))
  return torch.stack(losses).mean() if losses else m1.sum()*0

def encoder_fingerprint(model):
 h=hashlib.sha256()
 for name,module in [('image',model.image),('text',model.txt),('disease',model.disease)]:
  for key,value in module.state_dict().items():h.update((name+key).encode());h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()

class SourceGraph:
 def __init__(self,training_records,model):
  if any(r['split']!='train' for r in training_records):raise ValueError('memory must contain source training records only')
  self.records=copy.deepcopy(training_records);self.model=model;self.calls=0
  with torch.no_grad():self.emb={c['clause_id']:model.txt(ids(c['text'])).detach() for r in self.records for c in r['clauses']}
  self.encoder_version=encoder_fingerprint(model)
  self.version=hashlib.sha256((json.dumps(self.records,sort_keys=True)+self.encoder_version).encode()).hexdigest()
 def retrieve(self,q,P,query_patient,K=3,budget=24,alpha=.8):
  if encoder_fingerprint(self.model)!=self.encoder_version:raise RuntimeError('stale graph/embedding cache: rebuild after encoder changes')
  self.calls+=1
  eligible=[r for r in self.records if r['patient_id']!=query_patient]
  clauses=[c for r in eligible for c in r['clauses']];nodes=[];index={};edges=[]
  def node(key):
   if key not in index:index[key]=len(nodes);nodes.append(key)
   return index[key]
  for r in eligible:
   e=node(('exam',r['exam_id']))
   for c in r['clauses']:
    j=node(('clause',c['clause_id']));f=node(('fact',c['exam_id'],c['clause_id'],c['begin'],c['end']));concept=node(('concept',c['finding']));anatomy=node(('concept','anatomy',c['location']))
    edges.extend([(e,j),(j,f),(f,concept),(f,anatomy)])
  if not nodes:return [],{'patient_excluded_before_graph':True,'nodes':[],'column_sum_error':0.,'iterations':0,'actual_tokens':0}
  n=len(nodes);adj=torch.zeros(n,n)
  for a,b in edges:adj[a,b]+=1;adj[b,a]+=1
  seed=torch.zeros(n);fact_scores=[];clause_scores=[]
  qn=F.normalize(q.detach(),dim=0)
  for c in clauses:
   cosine=qn@F.normalize(self.emb[c['clause_id']],dim=0);state=c['assertion'];score=.5*cosine+.5*torch.log(P[c['finding'],state].detach()+1e-6) if state<3 else cosine
   fact_scores.append(score);clause_scores.append(cosine)
  fs=torch.stack(fact_scores);cs=torch.stack(clause_scores);top=torch.topk(fs,min(4,len(fs))).indices;fv=fs[top].softmax(0);cv=cs.softmax(0)*.5
  for rank,ci in enumerate(top.tolist()):
   c=clauses[ci];seed[index[('fact',c['exam_id'],c['clause_id'],c['begin'],c['end'])]]=fv[rank]
  for i,c in enumerate(clauses):seed[index[('clause',c['clause_id'])]]=cv[i]
  seed/=seed.sum();sums=adj.sum(0);T=adj/sums.clamp_min(1)[None];T[:,sums==0]=seed[:,None]
  pi=seed.clone();converged=False
  for iteration in range(100):
   nxt=(1-alpha)*seed+alpha*T@pi
   if float((nxt-pi).abs().sum())<1e-7:converged=True;pi=nxt;break
   pi=nxt
  ranks=sorted(range(len(clauses)),key=lambda i:(-float(pi[index[('clause',clauses[i]['clause_id'])]]) if converged else -float(cs[i]),clauses[i]['clause_id']))
  packed=[];tokens=0
  for i in ranks:
   needed=len(ids(clauses[i]['text']))+2
   if tokens+needed<=budget and len(packed)<K:packed.append(clauses[i]);tokens+=needed
  return packed,{'patient_excluded_before_graph':True,'eligible_patients':[r['patient_id'] for r in eligible],'nodes':[list(x) for x in nodes],'column_sum_error':float((T.sum(0)-1).abs().max()),'iterations':iteration+1,'converged':converged,'dense_fallback':not converged,'actual_tokens':tokens,'version':self.version}

def weak_support_target(candidate,target_relations):
 if not candidate['atomic']:return 2
 match=[r for r in target_relations if r['finding']==candidate['finding'] and r['location']==candidate['location']]
 if not match:return 2
 for r in match:
  if r['assertion']==2:return 2
  if candidate['assertion']==r['assertion'] and candidate['side']==r['side']:return 0
  if {candidate['assertion'],r['assertion']}=={0,1}:return 1
  if r['assertion']==0 and candidate['assertion']==0 and {candidate['side'],r['side']}=={0,1}:return 1
 return 2

def target_tokens(record):return ids(' '.join(c['text'] for c in record['clauses']))+[TOK['<eos>']]
def kept_positions(target,draft,clauses):
 positions=set()
 for c in clauses:
  seq=ids(c['text']);in_draft=any(draft[j:j+len(seq)]==seq for j in range(max(0,len(draft)-len(seq)+1)))
  if in_draft:
   for j in range(max(0,len(target)-len(seq)+1)):
    if target[j:j+len(seq)]==seq:positions.update(range(j,j+len(seq)))
 return sorted(positions)

def infer(model,graph,image,patient,max_tokens=14):
 with torch.no_grad():
  H,q,P=model.encode_image(image);clauses,record=graph.retrieve(q,P,patient);m0,w,p,states,lengths=model.context(H,q,P,clauses,False)
  draft=model.decoder.generate(m0,w,max_tokens);draft_features=model.txt.tokens(draft).detach();m1=model.correction(m0,w,draft_features);final=model.decoder.generate(m1,w,max_tokens)
 return {'draft_tokens':draft,'final_tokens':final,'draft':text(draft),'final':text(final),'candidate_ids':[c['clause_id'] for c in clauses],'candidate_patients':[c['patient_id'] for c in clauses],'support_probabilities':p.tolist(),'states':states,'graph':record,'retrievals':1,'memory_weight':w.tolist(),'image_token_count':len(H)}


# Stage A label helpers. These functions are training-only and are never called
# by SourceGraph.retrieve or infer; inference queries use predicted probabilities.
def relation_rows(clauses):
 """Flatten explicit relation records; do not infer any missing attribute."""
 rows=[]
 for clause in clauses:
  rows.extend(clause.get('relations',[clause]))
 return rows

def masked_assertion_loss(probabilities,clauses):
 """13 x 3 head loss, excluding unmentioned and ambiguous disease targets.

 Multiple observed facts for one disease contribute once only when their disease
 assertion agrees. Location-specific positive/negative ambiguity is masked; it
 must not be collapsed into a fabricated global negative or two opposing labels.
 """
 grouped={};unmentioned=0;outside=0
 for r in relation_rows(clauses):
  state=r.get('assertion',3);finding=r.get('finding')
  if state==3 or state is None:unmentioned+=1;continue
  if state not in (0,1,2):raise ValueError('unknown assertion code')
  if not isinstance(finding,int) or not 0<=finding<probabilities.shape[0]:outside+=1;continue
  grouped.setdefault(finding,set()).add(state)
 terms=[];ambiguous=0
 for finding,states in grouped.items():
  if len(states)!=1:ambiguous+=1;continue
  state=next(iter(states));terms.append(-probabilities[finding,state].clamp_min(1e-7).log())
 loss=torch.stack(terms).mean() if terms else probabilities.sum()*0
 return loss,{'supervised_heads':len(terms),'unmentioned_masked':unmentioned,'ambiguous_heads_masked':ambiguous,'outside_vocabulary_masked':outside}

def observed_fact_map(clauses):
 """Semantic key: finding, anatomical site, and laterality, never list position."""
 mapping={}
 for r in relation_rows(clauses):
  state=r.get('assertion',3)
  if state not in (0,1,2):continue
  # Missing keys remain unavailable rather than being aligned by list index.
  if any(r.get(k) is None for k in ('finding','location','side')):continue
  key=(r['finding'],r['location'],r['side'])
  mapping.setdefault(key,set()).add(state)
 return mapping

def assertion_agreement(query_clauses,candidate_clauses):
 """Nonnegative agreement on jointly observed facts, with contradiction veto.

 Present/absent at the SAME known relation key is an explicit contradiction.
 Left-present versus right-present is not itself evidence of mutual exclusion:
 bilateral disease is possible, so disjoint keys receive no agreement instead.
 Duplicate internally contradictory keys also receive zero, never a fabricated
 target. Unmentioned assertions and missing key fields cannot generate agreement.
 """
 a=observed_fact_map(query_clauses);b=observed_fact_map(candidate_clauses);keys=a.keys()&b.keys()
 score=0.
 for key in keys:
  aa,bb=a[key],b[key]
  if len(aa)!=1 or len(bb)!=1:return 0.
  left,right=next(iter(aa)),next(iter(bb))
  if {left,right}=={0,1}:return 0.
  score+=float(left==right)
 return score

def patient_aware_dm_loss(logits,queries,candidates):
 """Each column is an encoded candidate CLAUSE with an owning patient ID.

 Same-patient clauses are removed from the softmax denominator, including other
 examinations. Agreement targets are metadata-derived constants (no gradients).
 A query without positive agreement is omitted; an all-omitted batch returns a
 differentiable zero attached to both image/text similarity paths.
 """
 if logits.shape!=(len(queries),len(candidates)):raise ValueError('candidate alignment mismatch')
 terms=[];eligible_counts=[];positive_counts=[]
 for i,query in enumerate(queries):
  eligible=[j for j,c in enumerate(candidates) if c['patient_id']!=query['patient_id']]
  eligible_counts.append(len(eligible))
  agreement=logits.new_tensor([assertion_agreement(query['clauses'],[candidates[j]]) for j in eligible]).detach()
  positive_counts.append(int((agreement>0).sum()))
  if not eligible or not bool(agreement.sum()>0):continue
  target=agreement/agreement.sum()
  terms.append(-(target*F.log_softmax(logits[i,eligible],dim=0)).sum())
 loss=torch.stack(terms).mean() if terms else logits.sum()*0
 return loss,{'active_queries':len(terms),'omitted_queries':len(queries)-len(terms),'eligible_clause_counts':eligible_counts,'positive_agreement_counts':positive_counts}

def patient_aware_contrastive_loss(similarity,records):
 """Keep paired diagonal positives; other same-patient entries are not negatives."""
 n=len(records)
 if similarity.shape!=(n,n):raise ValueError('contrastive pair alignment mismatch')
 allowed=torch.tensor([[i==j or a['patient_id']!=b['patient_id'] for j,b in enumerate(records)] for i,a in enumerate(records)],device=similarity.device)
 masked=similarity.masked_fill(~allowed,-torch.inf);target=torch.arange(n,device=similarity.device)
 return (F.cross_entropy(masked,target)+F.cross_entropy(masked.T,target))/2
