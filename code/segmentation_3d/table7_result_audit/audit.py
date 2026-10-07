from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import sys, csv, json, re, math, hashlib, argparse
from pathlib import Path
import torch
sys.dont_write_bytecode=True
A=OUTPUT
B=A.parent.parent
R=RUN3
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1048576),b''):h.update(z)
    return h.hexdigest()
files=sorted(p for p in R.rglob('*') if p.is_file())
before={str(p):sha(p) for p in files}
(A/'hashes_before.json').write_text(json.dumps(before,indent=2))
def readcsv(n):
    with (R/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
h=readcsv('training_history.csv');p=readcsv('test_predictions.csv');t=readcsv('test_metrics.csv')[0]
c=json.loads((R/'run_config.txt').read_text(encoding='utf-8'))
with torch.serialization.safe_globals([argparse.Namespace]):
    best=torch.load(R/'best_model.pth',map_location='cpu',weights_only=True)
    last=torch.load(R/'last_checkpoint.pth',map_location='cpu',weights_only=True)
assert [int(x['epoch']) for x in h]==list(range(1,101))
assert last['epoch']==99
assert all(math.isfinite(float(v)) for row in h for v in row.values())
maximum=max(float(x['val_dice']) for x in h)
ties=[int(x['epoch']) for x in h if float(x['val_dice'])==maximum]
epoch=ties[0]
assert best['epoch']+1==epoch==int(t['best_epoch'])
assert best['best_val_dsc']==last['best_val_dsc']==maximum
assert best['training_config']==c==last['training_config']
blocks=re.findall(r'\[epoch: (\d+)\]\n(.*?)(?=\[epoch:|\Z)',(R/'training_log.txt').read_text(encoding='utf-8'),re.S)
assert len(blocks)==100
for row,(ep,block) in zip(h,blocks):
    assert int(row['epoch'])==int(ep)
    for key,label,fmt in [('train_loss','train_loss','.4f'),('val_dice','dice coefficient','.4f'),('val_miou','mIoU','.4f'),('val_ppv','PPV','.4f'),('lr','lr','.6f')]:
        saved=re.search(r'^'+re.escape(label)+r': ([^\n]+)',block,re.M).group(1)
        assert saved==format(float(row[key]),fmt),(ep,key,saved)
    detail=json.loads(next(line for line in block.splitlines() if line.startswith('{')))
    for key in ['PPV_volume_mean','ppv_num_volumes','ppv_zero_prediction_volumes']:
        assert float(detail[key])==float(row[key])
# Reconstruct ReduceLROnPlateau from saved training losses, without model/optimizer execution.
lr=c['lr'];loss_best=float('inf');bad=0;reductions=[]
for row in h:
    loss=float(row['train_loss'])
    if loss<loss_best:loss_best=loss;bad=0
    else:bad+=1
    if bad>c['lr_patience_internal']:
        new=lr*c['lr_factor']
        if lr-new>1e-8:lr=new;reductions.append(int(row['epoch']))
        bad=0
    assert abs(float(row['lr'])-lr)<1e-15
assert abs(last['optimizer']['param_groups'][0]['lr']-lr)<1e-15
assert last['lr_scheduler']['last_epoch']==100
assert last['lr_scheduler']['num_bad_epochs']==bad
assert last['lr_scheduler']['best']==loss_best
assert len(p)==int(t['ppv_num_volumes'])==c['test_count']==141
assert [int(x['index']) for x in p]==list(range(141))
assert len({x['image_path'] for x in p})==141
snapshot=(R/'code_snapshot/train3d.py').read_text(encoding='utf-8')
assert 'is_best = dice > best_dice' in snapshot
assert snapshot.index('model.load_state_dict(best_checkpoint["model"])')<snapshot.index('test_confmat, test_dice')<snapshot.index('test_records = save_test_predictions')
metric=(R/'code_snapshot/train_utils/segmentation3d_metrics.py').read_text(encoding='utf-8')
assert 'ignore_index=255, epsilon=1e-6' in metric
dataset=(R/'code_snapshot/my_dataset3d.py').read_text(encoding='utf-8')
assert 'mask = (mask > 0).astype(np.int64)' in dataset
total_voxels=c['depth']*c['input_size']**2
counts=[];ds32=[]
for row in p:
    gt=int(row['gt_positive_pixels']);pred=int(row['pred_positive_pixels'])
    estimate=float(row['ppv'])*pred;tp=round(estimate)
    assert abs(estimate-tp)<1e-7 and 0<=tp<=min(gt,pred)
    fp=pred-tp;fn=gt-tp;tn=total_voxels-tp-fp-fn
    assert tn>=0
    assert abs(float(row['ppv'])-(tp/pred if pred else 0))<1e-14
    assert abs(float(row['dice'])-(2*tp/(gt+pred) if gt+pred else 1))<1e-14
    assert abs(float(row['iou'])-(tp/(tp+fp+fn) if tp+fp+fn else 1))<1e-14
    x=torch.tensor(float(tp),dtype=torch.float32);den=torch.tensor(float(gt+pred),dtype=torch.float32)
    ds32.append(((2*x+1e-6)/(den+1e-6)).item())
    counts.append(dict(index=int(row['index']),TP=tp,TN=tn,FP=fp,FN=fn,total_voxels=total_voxels))
TP=sum(x['TP'] for x in counts);TN=sum(x['TN'] for x in counts);FP=sum(x['FP'] for x in counts);FN=sum(x['FN'] for x in counts)
cm=torch.tensor([[TN,FP],[FN,TP]],dtype=torch.int64);hf=cm.float()
iu=hf.diag()/(hf.sum(0)+hf.sum(1)-hf.diag())
miou=iu.mean().item()
ppvmean=sum(float(x['ppv']) for x in p)/len(p)
dsmean=sum(float(x['dice']) for x in p)/len(p);dsformal=sum(ds32)/len(ds32)
zero=sum(int(x['pred_positive_pixels'])==0 for x in p)
assert zero==int(t['ppv_zero_prediction_volumes'])==5
assert ppvmean==float(t['PPV_volume_mean'])==float(t['PPV'])==float(t['test_ppv'])
assert abs(dsmean-float(t['test_dice']))<1e-6
assert dsformal==float(t['test_dice'])
assert miou==float(t['test_miou'])
checks=[]
for name,value,key in [('CSV DSC mean (no epsilon)',dsmean,'test_dice'),('Reconstructed DSC (epsilon/float32)',dsformal,'test_dice'),('PPV_volume_mean',ppvmean,'PPV_volume_mean'),('mIoU float32',miou,'test_miou')]:
    checks.append(dict(Metric=name,Recomputed=value,Official=float(t[key]),Absolute_difference=abs(value-float(t[key]))))
comparison=[dict(Metric=name,Paper_percent=paper,Reproduction_percent=float(t[key])*100,Difference_pp=float(t[key])*100-paper) for name,paper,key in [('DSC',66.5,'test_dice'),('mIoU',75.1,'test_miou'),('PPV',83.3,'PPV_volume_mean')]]
evidence=[]
def ev(source,ep='N/A',val='N/A',d='N/A',j='N/A',v='N/A',status='一致'):
    evidence.append(dict(Evidence_source=source,Best_Epoch=ep,Best_Val_DSC=val,Test_DSC=d,Test_mIoU=j,Test_PPV=v,Status=status))
ev('history',epoch,maximum)
ev('training log',epoch,format(maximum,'.4f'),status='一致（DSC日志四位小数）')
ev('best checkpoint',epoch,best['best_val_dsc'])
ev('test_metrics.csv',epoch,d=t['test_dice'],j=t['test_miou'],v=t['PPV_volume_mean'])
ev('test_predictions.csv (derived)',d=dsmean,j=miou,v=ppvmean,status='一致（DSC epsilon/浮点差异；mIoU结合保存配置推导）')
def writecsv(name,rows):
    with (A/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('paper_vs_reproduction.csv',comparison);writecsv('recomputed_metric_check.csv',checks);writecsv('evidence_chain.csv',evidence);writecsv('reconstructed_counts.csv',counts)
summary=dict(epochs_complete=True,best_epoch=epoch,best_checkpoint_raw_epoch=best['epoch'],last_checkpoint_raw_epoch=last['epoch'],best_validation_DSC=maximum,tied_max_epochs=ties,formal_test={key:t[key] for key in ('best_epoch','test_dice','test_miou','test_ppv','PPV','PPV_volume_mean','ppv_num_volumes','ppv_zero_prediction_volumes')},recomputations=checks,confusion_matrix=cm.tolist(),class_IoUs=iu.tolist(),lr_reduction_epochs=reductions,final_lr=lr,nan_inf_in_saved_metrics=False,paper_comparison=comparison,checkpoint_optimizer_weight_decay=last['optimizer']['param_groups'][0]['weight_decay'],formal_files=len(files),evidence_chain='一致')
(A/'audit_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
after={str(p):sha(p) for p in files};assert before==after
assert files==sorted(p for p in R.rglob('*') if p.is_file())
(A/'hashes_after.json').write_text(json.dumps(after,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
