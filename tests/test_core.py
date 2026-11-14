import unittest,sys,json,copy,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.core import *
from src.data import generate
torch.set_num_threads(1)
class Contracts(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);self.m=Model();self.train=generate(None)['records'][:10]
 def test_exclusion_precedes_graph_and_column_normalized(self):
  g=SourceGraph(self.train,self.m);H,q,P=self.m.encode_image(torch.rand(8,8));clauses,a=g.retrieve(q,P,0)
  self.assertNotIn(0,a['eligible_patients']);self.assertTrue(all(c['patient_id']!=0 for c in clauses));self.assertTrue(all('p0_c' not in str(n) and 'exam_0' not in str(n) for n in a['nodes']));self.assertLess(a['column_sum_error'],1e-6);self.assertLessEqual(a['actual_tokens'],24)
 def test_nontraining_memory_rejected(self):
  records=copy.deepcopy(self.train);records[0]['split']='test'
  with self.assertRaises(ValueError):SourceGraph(records,self.m)
 def test_conflict_priority_composite_never_supported(self):
  p=torch.tensor([[.7,.5,.1],[.9,.05,.05],[.9,.05,.05]])
  w,s=hard_gate(p,torch.tensor([True,False,True]));self.assertEqual(s.tolist(),[1,2,0]);self.assertEqual(float(w[0]),0);self.assertAlmostEqual(float(w[1]),.135,places=6)
 def test_weight_after_projection_and_mask_survives_residual(self):
  q=torch.randn(3,8);mem=torch.randn(5,8);lin=[nn.Linear(8,8) for _ in range(3)];w=torch.ones(5)
  a=projected_attention(q,mem,w,*lin);b=projected_attention(q,mem,.2*w,*lin);self.assertTrue(torch.allclose(b,.2*a,atol=1e-6))
  w[-1]=0;other=mem.clone();other[-1]+=100;self.assertTrue(torch.allclose(projected_attention(q,mem,w,*lin),projected_attention(q,other,w,*lin)))
 def test_frozen_decoder_preserves_adapter_input_gradients(self):
  freeze(self.m,True);freeze(self.m.correction,False);m0=torch.randn(8,8);w=torch.ones(8);w[-2:]=0;draft=torch.randn(3,8);m1=self.m.correction(m0,w,draft)
  loss,_=self.m.decoder.teacher(ids('left opacity lung present .')+[TOK['<eos>']],m1,w);loss.backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.m.correction.parameters()),0);self.assertTrue(all(p.grad is None for p in self.m.decoder.parameters()));self.assertTrue(all(p.grad is None for p in self.m.image.parameters()))
 def test_one_retrieval_two_pass_inference_and_frozen_state(self):
  freeze(self.m,True);g=SourceGraph(self.train,self.m);before={k:v.clone() for k,v in self.m.state_dict().items()};out=infer(self.m,g,torch.rand(8,8),99,5);self.assertEqual(g.calls,1);self.assertEqual(out['retrievals'],1);self.assertTrue(all(torch.equal(before[k],v) for k,v in self.m.state_dict().items()))
 def test_keep_loss_only_complete_correct_draft_clause(self):
  c=self.train[0]['clauses'];target=target_tokens(self.train[0]);self.assertEqual(kept_positions(target,ids('left .'),c),[]);full=ids(c[0]['text']);self.assertEqual(len(kept_positions(target,full,c)),len(full))
 def test_same_finding_conflicting_attributes_masked(self):
  r=copy.deepcopy(self.train[0]['clauses'][0]);other=copy.deepcopy(r);other['side']=1-r['side'];other['assertion']=(r['assertion']+1)%3;m=torch.randn(7,8,requires_grad=True)
  loss=self.m.relation_loss(m,torch.ones(7),[r,other]);self.assertEqual(float(loss.detach()),0)
 def test_unmentioned_not_absent(self):
  c=copy.deepcopy(self.train[0]['clauses'][0]);c['assertion']=3;self.assertEqual(weak_support_target(c,self.train[0]['clauses']),2)
 def test_stale_encoder_cache_fails_explicitly(self):
  graph=SourceGraph(self.train,self.m)
  with torch.no_grad():self.m.txt.embed.weight.add_(.01)
  H,q,P=self.m.encode_image(torch.rand(8,8))
  with self.assertRaises(RuntimeError):graph.retrieve(q,P,99)
 def test_stage_a_unmentioned_mask_and_differentiable_empty_classification(self):
  raw=torch.randn(13,3,requires_grad=True);p=raw.softmax(-1);c=copy.deepcopy(self.train[0]['clauses'][0]);c['assertion']=3
  loss,audit=masked_assertion_loss(p,[c]);self.assertEqual(float(loss.detach()),0);self.assertEqual(audit['supervised_heads'],0);self.assertEqual(audit['unmentioned_masked'],1)
  loss.backward();self.assertTrue(torch.equal(raw.grad,torch.zeros_like(raw)))
 def test_stage_a_observed_assertion_only_and_conflicting_head_mask(self):
  raw=torch.randn(13,3,requires_grad=True);p=raw.softmax(-1);c=copy.deepcopy(self.train[0]['clauses'][0]);c['assertion']=0;unknown=copy.deepcopy(c);unknown['assertion']=3
  loss,audit=masked_assertion_loss(p,[unknown,c]);self.assertTrue(torch.allclose(loss,-p[c['finding'],0].log()));self.assertEqual(audit['supervised_heads'],1)
  conflict=copy.deepcopy(c);conflict['assertion']=1;zero,report=masked_assertion_loss(p,[c,conflict]);self.assertEqual(float(zero.detach()),0);self.assertEqual(report['ambiguous_heads_masked'],1)
 def test_stage_a_same_patient_other_exam_removed_from_dm_denominator(self):
  query=copy.deepcopy(self.train[0]);c=copy.deepcopy(query['clauses'][0]);c['assertion']=0;query['clauses']=[c]
  own=dict(c,patient_id=query['patient_id'],exam_id='another_exam_same_patient');other=dict(c,patient_id=999);unknown=dict(other,patient_id=998,assertion=3)
  logits=torch.tensor([[100.,1.,-1.]],requires_grad=True);loss,audit=patient_aware_dm_loss(logits,[query],[own,other,unknown]);expected=-F.log_softmax(logits[0,1:],0)[0]
  self.assertTrue(torch.allclose(loss,expected));self.assertEqual(audit['eligible_clause_counts'],[2]);loss.backward();self.assertEqual(float(logits.grad[0,0]),0);self.assertGreater(float(logits.grad[0,1:].abs().sum()),0)
 def test_stage_a_fact_key_order_independent_and_contradiction_veto(self):
  clauses=copy.deepcopy(self.train[0]['clauses']);clauses[0]['assertion']=0;clauses[1]['assertion']=1
  self.assertEqual(assertion_agreement(clauses,list(reversed(clauses))),2)
  changed=copy.deepcopy(clauses);changed[0]['assertion']=1
  # One matching fact must not hide an explicit present/absent contradiction.
  self.assertEqual(assertion_agreement(clauses,changed),0)
  swapped_keys=copy.deepcopy(clauses);swapped_keys[0]['finding']=9;swapped_keys[1]['finding']=10
  self.assertEqual(assertion_agreement(clauses,swapped_keys),0)
 def test_stage_a_unmentioned_never_agreement_and_all_zero_dm_gradients(self):
  q=copy.deepcopy(self.train[0]);c=copy.deepcopy(q['clauses'][0]);c['assertion']=3;q['clauses']=[c];other=dict(c,patient_id=999)
  self.assertEqual(assertion_agreement([c],[other]),0);logits=torch.randn(1,1,requires_grad=True);loss,audit=patient_aware_dm_loss(logits,[q],[other]);self.assertEqual(audit['active_queries'],0);self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))
 def test_stage_a_no_other_patient_candidates_returns_differentiable_zero(self):
  q=copy.deepcopy(self.train[0]);logits=torch.randn(1,1,requires_grad=True);loss,audit=patient_aware_dm_loss(logits,[q],[q['clauses'][0]]);self.assertEqual(audit['eligible_clause_counts'],[0]);loss.backward();self.assertEqual(float(logits.grad.abs().sum()),0)
  empty=torch.empty(1,0,requires_grad=True);loss,audit=patient_aware_dm_loss(empty,[q],[]);loss.backward();self.assertIsNotNone(empty.grad)
 def test_stage_a_same_patient_not_contrastive_negative(self):
  records=[{'patient_id':0},{'patient_id':0},{'patient_id':1}];sim=torch.randn(3,3,requires_grad=True);loss=patient_aware_contrastive_loss(sim,records);loss.backward();self.assertEqual(float(sim.grad[0,1]),0);self.assertEqual(float(sim.grad[1,0]),0);self.assertGreater(float(sim.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
