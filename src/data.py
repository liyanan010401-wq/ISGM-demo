import json
from pathlib import Path
import numpy as np

def generate(path,seed=4404):
 rng=np.random.default_rng(seed);records=[]
 for i in range(15):
  image=rng.normal(.1,.015,(8,8)).astype(np.float32);clauses=[]
  for finding in (0,1):
   side=(i+finding)%2;assertion=(i//2+finding)%3;x=2 if side==0 else 5;y=2+3*finding
   if assertion==0:image[y-1:y+2,x-1:x+2]+=.6
   elif assertion==2:image[y-1:y+2,x-1:x+2]+=.2
   words=['left' if side==0 else 'right',['opacity','effusion'][finding],'lung',['present','absent','uncertain'][assertion],'.'];clause=' '.join(words)
   clauses.append({'patient_id':i,'exam_id':'exam_'+str(i),'clause_id':f'p{i}_c{finding}','begin':0,'end':len(clause),'finding':finding,'location':'lung','side':side,'assertion':assertion,'atomic':True,'text':clause})
  records.append({'patient_id':i,'exam_id':'exam_'+str(i),'split':'train' if i<10 else 'dev' if i<12 else 'test','image':image.tolist(),'clauses':clauses})
 payload={'status':'fully synthetic nonmedical records','seed':seed,'records':records}
 if path is not None:Path(path).write_text(json.dumps(payload,indent=2),encoding='utf-8',newline='\n')
 return payload
if __name__=='__main__':generate(Path(__file__).resolve().parents[1]/'data/sample.json')
