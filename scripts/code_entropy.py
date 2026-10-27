"""Per-codebook index entropy on a frozen window bank -> effective bits/s,
plus the temporal and cross-stage redundancy an entropy coder could exploit.

Usage: python scripts/code_entropy.py --inputs BANK.npz --output entropy.csv RUN:LABEL [RUN:LABEL ...]

Per codebook i (frames t, index c):
  H_i          marginal entropy H(c_t)                       -> effective bits/s (as before)
  Hc_i         first-order conditional entropy H(c_t | c_{t-1}), plug-in estimate
  Hc_shuf_i    the same on a frame-shuffled sequence: same marginal and same table
               sparsity, but no true temporal dependence. Its deficit below H_i is pure
               estimator bias, so
  MI_i         = Hc_shuf_i - Hc_i  is the bias-corrected mutual information
               between adjacent frames (bits recoverable by a 1-frame-memory coder)
  Hx_i         H(c_i,t | c_0,t): stage i given stage 0 in the same frame (RVQ only, i>0)
  Hx_shuf_i    shuffle control for Hx_i;  MIx_i = Hx_shuf_i - Hx_i
Totals:
  effective_bits_per_s           frame_rate * sum_i H_i          (marginal, as before)
  conditional_bits_per_s         frame_rate * sum_i (H_i - MI_i) (1-frame-memory coder)
  temporal_redundancy_fraction   sum_i MI_i / sum_i H_i
Window-edge frames (--trim samples each side) are dropped before counting; pairs never
span two windows.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'scripts'))
from dac.model import DAC
from evaluate_eeg import resolve_checkpoint

p=argparse.ArgumentParser(); p.add_argument('runs',nargs='+'); p.add_argument('--inputs',required=True); p.add_argument('--output',required=True)
p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--trim',type=int,default=128)
p.add_argument('--shuffle-seed',type=int,default=0); p.add_argument('--save-codes',action='store_true',help='also write <output stem>_<label>_codes.npy (B x N x T int16)')
a=p.parse_args()


def entropy_bits(ids):
    """Plug-in Shannon entropy (bits) of a 1-D integer tensor."""
    _,cnt=torch.unique(ids,return_counts=True); pr=cnt.double()/cnt.sum()
    return float(-(pr*pr.log2()).sum())


def conditional_entropy(cond,target,K):
    """H(target | cond) = H(cond,target) - H(cond) for 1-D integer tensors with values < K."""
    joint=cond.long()*K+target.long()
    return entropy_bits(joint)-entropy_bits(cond)


x=np.load(a.inputs)['audio'].astype(np.float32); rate=256
gen=torch.Generator().manual_seed(a.shuffle_seed)
rows=[]
for spec in a.runs:
    run,label=spec.split(':') if ':' in spec else (spec,Path(spec).name)
    model=DAC.load(str(resolve_checkpoint(run)),strict=True).to(a.device).eval()
    hop=int(np.prod(model.encoder_rates)); K=model.codebook_size; n=model.n_codebooks
    codes=[]
    with torch.inference_mode():
        for s in range(0,len(x),a.batch_size):
            b=torch.from_numpy(x[s:s+a.batch_size,None]).to(a.device)
            _,c,_,_,_=model.encode(model.preprocess(b,rate)); codes.append(c.cpu())
    codes=torch.cat(codes,0)  # B x N x T
    T=codes.shape[-1]; keep=slice(a.trim//hop, T-a.trim//hop)  # drop frames in the trimmed 0.5 s edges
    codes=codes[:,:,keep]
    B,_,T=codes.shape
    if a.save_codes:
        np.save(Path(a.output).with_name(f"{Path(a.output).stem}_{label}_codes.npy"),codes.numpy().astype(np.int16))

    row=dict(label=label,run=run,hop=hop,n_codebooks=n,codebook_size=K,frame_rate=rate/hop,
             windows=B,frames_per_window=T,pairs=B*(T-1),nominal_bits_per_s=rate/hop*n*np.log2(K))
    H,MI=[],[]
    for i in range(n):
        seq=codes[:,i]                                   # B x T
        H_i=entropy_bits(seq.flatten())
        # --- temporal: adjacent frames within a window
        prev,nxt=seq[:,:-1].flatten(),seq[:,1:].flatten()
        Hc=conditional_entropy(prev,nxt,K)
        # shuffle control: permute frames across the whole bank, then re-pair.
        perm=torch.randperm(seq.numel(),generator=gen)
        shuf=seq.flatten()[perm].view(B,T)
        Hc_s=conditional_entropy(shuf[:,:-1].flatten(),shuf[:,1:].flatten(),K)
        MI_i=Hc_s-Hc
        H.append(H_i); MI.append(MI_i)
        row.update({f'H_{i}':H_i,f'Hc_{i}':Hc,f'Hc_shuf_{i}':Hc_s,f'MI_{i}':MI_i})
        # --- cross-stage: stage i given stage 0, same frame (RVQ only)
        if i>0:
            c0=codes[:,0].flatten(); ci=seq.flatten()
            Hx=conditional_entropy(c0,ci,K)
            Hx_s=conditional_entropy(c0,ci[torch.randperm(ci.numel(),generator=gen)],K)
            row.update({f'Hx_{i}':Hx,f'Hx_shuf_{i}':Hx_s,f'MIx_{i}':Hx_s-Hx})
    row['effective_bits_per_s']=rate/hop*sum(H)
    row['conditional_bits_per_s']=rate/hop*sum(h-m for h,m in zip(H,MI))
    row['temporal_redundancy_fraction']=sum(MI)/sum(H)
    rows.append(row)
    print(f"{label} hop{hop} {n}x{K}: H={[round(h,2) for h in H]} MI={[round(m,3) for m in MI]} "
          f"nominal={row['nominal_bits_per_s']:.0f} effective={row['effective_bits_per_s']:.1f} "
          f"conditional={row['conditional_bits_per_s']:.1f} bits/s  redundancy={100*row['temporal_redundancy_fraction']:.1f}%",flush=True)
    del model; torch.cuda.empty_cache()

fixed=['label','run','hop','n_codebooks','codebook_size','frame_rate','windows','frames_per_window','pairs',
       'nominal_bits_per_s','effective_bits_per_s','conditional_bits_per_s','temporal_redundancy_fraction']
keys=fixed+sorted({k for r in rows for k in r}-set(fixed))
with open(a.output,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
