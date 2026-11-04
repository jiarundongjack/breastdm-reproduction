"""Reconstruct aggregate segmentation metrics from saved per-sample statistics.

Requires the audited binary, fixed-size, no-ignored-voxel protocol. Does not
load images, weights, patient masks, torch pickles, or experiment modules.
"""
import argparse, csv, hashlib, json, math, struct
from pathlib import Path

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def reconstruct(rows, mode, total):
    counts=[]; dice_values=[]; eligible=[]; ppvs=[]; cumulative=f32(0)
    if not rows: raise ValueError('Empty predictions')
    if [int(r['index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Expected original contiguous sample order starting at zero')
    for r in rows:
        gt=int(r['gt_positive_pixels']); pred=int(r['pred_positive_pixels'])
        ppv=float(r['ppv']); estimate=ppv*pred
        if not math.isfinite(estimate): raise ValueError('Non-finite count estimate')
        tp=round(estimate); fp=pred-tp; fn=gt-tp; tn=total-tp-fp-fn
        if abs(estimate-tp)>1e-7 or min(tp,fp,fn,tn)<0:
            raise ValueError('Counts cannot be reconstructed under selected protocol')
        expected={'ppv':tp/pred if pred else 0,
                  'dice':2*tp/(gt+pred) if gt+pred else 1,
                  'iou':tp/(tp+fp+fn) if tp+fp+fn else 1}
        for key,value in expected.items():
            observed=float(r[key])
            if not math.isfinite(observed) or abs(observed-value)>1e-12:
                raise ValueError(f'Inconsistent saved {key}')
        d=f32(f32(f32(2*tp)+f32(1e-6))/f32(f32(gt+pred)+f32(1e-6)))
        dice_values.append(d); cumulative=f32(cumulative+d)
        counts.append((tn,fp,fn,tp));ppvs.append(ppv)
        if tp>0 and tn>0:eligible.append(ppv)
    tn,fp,fn,tp=map(sum,zip(*counts))
    a,b,c,d=map(f32,(tn,fp,fn,tp))
    # Match float32 confusion-matrix row/column operations in saved evaluator.
    bgden=f32(f32(f32(a+c)+f32(a+b))-a)
    fgden=f32(f32(f32(b+d)+f32(c+d))-d)
    miou=f32(f32(f32(a/bgden)+f32(d/fgden))/2)
    global_ppv=f32(d/f32(d+b)) if d+b else 0
    metrics={'test_dice':f32(cumulative/f32(len(rows))) if mode=='2d' else sum(dice_values)/len(rows),
             'test_miou':miou,
             'test_ppv':global_ppv if mode=='2d' else sum(ppvs)/len(rows)}
    if mode=='2d':
        metrics.update(PPV_global=global_ppv,PPV_author_separate_evaluator=sum(eligible)/len(eligible),
                       author_ppv_included_images=len(eligible),author_ppv_excluded_images=len(rows)-len(eligible))
    else:
        metrics.update(PPV_volume_mean=sum(ppvs)/len(rows),PPV_global_aux=global_ppv,
                       ppv_num_volumes=len(rows),ppv_zero_prediction_volumes=sum(int(r['pred_positive_pixels'])==0 for r in rows))
    return metrics,[[tn,fp],[fn,tp]]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['2d','3d'],required=True)
    p.add_argument('--spatial-size',type=int,required=True)
    p.add_argument('--depth',type=int,default=1)
    p.add_argument('--confirm-no-ignored-voxels',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.spatial_size<=0 or a.depth<=0 or (a.mode=='2d' and a.depth!=1):p.error('Invalid dimensions')
    if a.output.exists():p.error('Output must be a new directory')
    paths=[a.run_dir/'test_predictions.csv',a.run_dir/'test_metrics.csv']
    raw=[q.read_bytes() for q in paths]
    hashes=[hashlib.sha256(b).hexdigest() for b in raw]
    import io
    rows=list(csv.DictReader(io.StringIO(raw[0].decode('utf-8-sig'))))
    official=next(csv.DictReader(io.StringIO(raw[1].decode('utf-8-sig'))))
    metrics,cm=reconstruct(rows,a.mode,a.spatial_size**2*a.depth)
    errors={k:abs(v-float(official[k])) for k,v in metrics.items()}
    exact=all(e==0 for e in errors.values())
    if not all(math.isfinite(e) and e<=1e-7 for e in errors.values()):
        raise ValueError(f'Metric verification failed: {errors}')
    if any(hashlib.sha256(q.read_bytes()).hexdigest()!=h for q,h in zip(paths,hashes)):
        raise ValueError('Input changed during verification')
    report=dict(mode=a.mode,samples=len(rows),spatial_size=a.spatial_size,depth=a.depth,
                assumption='Fixed-size binary masks, no ignored voxels; confirmed from formal loader/evaluator before invocation.',
                method='Integer TP reconstructed from saved PPV times predicted-positive count; validate against saved DSC/IoU. Float32 evaluator aggregation emulated. 2D DSC sequential float32 sum, 3D mean of float32 per-volume DSC; global two-class mIoU.',
                metrics=metrics,confusion_matrix=cm,absolute_errors=errors,exact_match=exact,
                verified_tolerance=1e-7,input_hashes_unchanged=True,
                inputs=[dict(path=str(q.resolve()),sha256=h) for q,h in zip(paths,hashes)],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Reconstruction from saved statistics; not independent mask evaluation, inference, training, or evidence of original published evaluation provenance.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(mode=a.mode,samples=len(rows),metrics=metrics,absolute_errors=errors,exact_match=exact),indent=2))

if __name__=='__main__':main()
