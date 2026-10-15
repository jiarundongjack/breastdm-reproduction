from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import sys, json, ast, warnings, csv, os, textwrap
from pathlib import Path
sys.dont_write_bytecode=True
A=OUTPUT; B=RUNTIME
sys.path.insert(0,str(B))
import torch
import numpy as np
from train_utils.segmentation3d_metrics import VolumeMetrics, METRIC_PROTOCOL
from my_dataset3d import standardize_depth
# Four synthetic volumes; includes complete miss, empty prediction, both empty.
g=torch.tensor([[1,1,0,0],[1,0,0,0],[0,0,0,0],[1,1,1,0]]).reshape(4,1,2,2)
pred=torch.tensor([[1,0,1,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]).reshape_as(g)
logits=torch.stack([(pred==0).float(),(pred==1).float()],1)
results=[]
for sizes in [[1,1,1,1],[3,1],[4]]:
 m=VolumeMetrics();start=0
 for n in sizes:m.update(logits[start:start+n],g[start:start+n]);start+=n
 results.append(m.compute())
for r in results[1:]:assert r[1:]==results[0][1:]
r=results[0];d=r[4]
assert d['PPV_volume_mean']==(.5+0+0+.75)/4
assert d['ppv_num_volumes']==4 and d['ppv_zero_prediction_volumes']==2
expected_dice=sum([(2+1e-6)/(4+1e-6),1e-6/(1+1e-6),1.,(6+1e-6)/(7+1e-6)])/4
assert abs(r[1]-expected_dice)<1e-7
cm=r[0].mat;h=cm.float();iou=h.diag()/(h.sum(0)+h.sum(1)-h.diag());assert r[2]==iou.mean().item()
# Missing tumor remains NaN, never filtered. Both-empty Dice=1, PPV=0.
m=VolumeMetrics();empty=torch.zeros((1,1,2,2),dtype=torch.long)
m.update(torch.stack([torch.ones_like(empty),torch.zeros_like(empty)],1).float(),empty)
with warnings.catch_warnings(record=True) as w:
 e=m.compute();assert np.isnan(e[2]) and w
assert e[1]==1 and e[3]==0
# Ignore label exclusion, and GT=0 padding contributes false positives.
m=VolumeMetrics();gt=torch.tensor([[[[1,255],[0,0]]]]);pr=torch.ones_like(gt)
m.update(torch.stack([(pr==0).float(),pr.float()],1),gt);q=m.compute();assert q[4]['PPV']==1/3 and int(q[0].mat.sum())==3
for n in [5,8,11]:
 x=np.broadcast_to(np.arange(n),(2,2,n)).copy();y=x.copy();xx,yy=standardize_depth(x,y)
 assert xx.shape==(2,2,8) and np.array_equal(xx,yy)
 if n==8:assert xx is x
 if n==11:assert np.array_equal(xx,x[:,:,1:9])
 if n==5:assert np.array_equal(xx[:,:,1:6],x) and not xx[:,:,:1].any() and not xx[:,:,6:].any()
# Real scheduler only, no optimizer step or model training.
a=torch.tensor(0.,requires_grad=True);opt=torch.optim.Adam([a],lr=.001)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.1,patience=9,threshold=0.)
sched.step(1.)
for _ in range(9):sched.step(1.)
assert opt.param_groups[0]['lr']==.001
sched.step(1.);assert opt.param_groups[0]['lr']==.0001
s=(B/'train3d.py').read_text(encoding='utf-8');ast.parse(s)
assert 'is_best = dice > best_dice' in s and '"best_val_dsc": best_dice' in s
# Actual formal CSV block, synthetic values written only in audit.
start=s.index('    test_metrics_file = os.path.join(run_dir, "test_metrics.csv")');end=s.index('    # 保存test集中',start)
sc=dict(os=os,csv=csv,run_dir=str(A),best_epoch=1,test_dice=r[1],test_miou=r[2],test_ppv=r[3],test_ppv_details=d)
exec(compile(textwrap.dedent(s[start:end]),'synthetic_csv_writer','exec'),sc)
row=next(csv.DictReader((A/'test_metrics.csv').open()));assert float(row['PPV'])==float(row['test_ppv'])==float(row['PPV_volume_mean'])==d['PPV']
# File/header inspection only; no data loader iteration or inference.
root=Path((SEG3_DATA));sets={};data={}
for split in ['train','val','test']:
 files=list((root/split/'images').rglob('*.npy'));sets[split]={p.relative_to(root/split/'images').parts[0] for p in files}
 for f in files:
  x=np.load(f,mmap_mode='r');y=np.load(root/split/'labels'/f.relative_to(root/split/'images'),mmap_mode='r');assert x.shape==y.shape==(369,369,8)
 data[split]={'volumes':len(files),'patients':len(sets[split])}
assert not any(sets[x]&sets[y] for x,y in [('train','val'),('train','test'),('val','test')])
report={'synthetic_batch_invariance':True,'all_volume_PPVs_included':True,'missing_class_NaN_retained':True,'depth_cases_5_8_11_pass':True,'scheduler_reduces_on_bad_epoch_10':True,'formal_CSV_writer_pass':True,'synthetic_DSC':r[1],'synthetic_PPV':d,'dataset':data,'patient_overlap':False,'training':False,'full_inference':False}
(A/'test_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
