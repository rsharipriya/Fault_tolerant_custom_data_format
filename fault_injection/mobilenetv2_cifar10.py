"""
unified_ptq_fault_robustness_mobilenetv2_cifar10.py  [FINAL]
═══════════════════════════════════════════════════════════════════════════════
PTQ-WOQ fault-robustness  —  MobileNetV2 / CIFAR-10

KEY CHANGES:
  1. Fault injection uses the FULL 10,000-sample test set (not 1,000 subset)
  2. Early-skip: if PTQ accuracy drop > ACCURACY_DROP_THRESHOLD (30%) below
     FP32 baseline, fault injection is skipped. Format still shown in bar
     chart with hatching.

Model adaptations for 32×32 CIFAR-10 images
─────────────────────────────────────────────
  features[0][0] : stride 2 → 1  (prevents spatial collapse on 32×32)
  classifier[1]  : Linear(1280, 10)
  PTH            : mobilenetv2_cifar10.pth
═══════════════════════════════════════════════════════════════════════════════
"""

import copy, json, math
import numpy as np
import torch, torch.nn as nn
import matplotlib, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

try:
    import cupy as cp
    from cupy import RawModule
    CUPY_OK = True
except ImportError:
    CUPY_OK = False
    print("[WARN] CuPy not found – using NumPy CPU fallback.")

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif","Georgia"],
    "mathtext.fontset":"cm","font.size":8,"axes.titlesize":8,"axes.labelsize":7,
    "xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":5.5,
    "legend.framealpha":0.88,"legend.edgecolor":"0.75","axes.linewidth":0.65,
    "xtick.major.width":0.55,"ytick.major.width":0.55,"xtick.direction":"in",
    "ytick.direction":"in","xtick.top":True,"ytick.right":True,
    "lines.linewidth":1.15,"lines.markersize":3.0,"axes.grid":True,
    "grid.linewidth":0.30,"grid.alpha":0.30,"grid.linestyle":"--",
    "figure.dpi":130,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
})
_T = plt.cm.tab20.colors
FMT_COLOR = {
    "E2M13":_T[0],"E3M12":_T[2],"E4M11":_T[4],"E5M10":_T[6],"E6M9":_T[8],
    "E7M8":_T[10],"E8M7":_T[12],"F8E3M4":_T[14],"F8E4M3":_T[16],"F8E5M2":_T[18],
    "INT16":_T[1],"INT8":_T[3],"INT4":_T[5],"INT2":_T[7],
}
FMT_LS = {
    **{f:"-"  for f in ["E2M13","E3M12","E4M11","E5M10","E6M9","E7M8","E8M7"]},
    **{f:"--" for f in ["F8E3M4","F8E4M3","F8E5M2"]},
    **{f:":"  for f in ["INT16","INT8","INT4","INT2"]},
}
FMT_MK = {
    **{f:"o"  for f in ["E2M13","E3M12","E4M11","E5M10","E6M9","E7M8","E8M7"]},
    **{f:"s"  for f in ["F8E3M4","F8E4M3","F8E5M2"]},
    **{f:"^"  for f in ["INT16","INT8","INT4","INT2"]},
}

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PTH_PATH     = "mobilenetv2_cifar10.pth"
NUM_CLASSES  = 10
DATASET_NAME = "CIFAR-10"
MODEL_NAME   = "MobileNetV2"
OUT_PREFIX   = "ptq_unified_mobilenetv2_cifar10"
BATCH_SIZE   = 256
N_TRIALS     = 100
N_FAULTS     = list(range(1, 9))

# ── EARLY-SKIP THRESHOLD ──────────────────────────────────────────────────────
ACCURACY_DROP_THRESHOLD = 30.0   # skip fault injection if PTQ drop > 30% from FP32

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

FLOAT_FORMATS = {
    "E2M13":(2,13,16),"E3M12":(3,12,16),"E4M11":(4,11,16),
    "E5M10":(5,10,16),"E6M9":(6,9,16),"E7M8":(7,8,16),"E8M7":(8,7,16),
    "F8E3M4":(3,4,8),"F8E4M3":(4,3,8),"F8E5M2":(5,2,8),
}
INT_FORMATS  = {"INT16":16,"INT8":8,"INT4":4,"INT2":2}
ALL_FMTS     = list(FLOAT_FORMATS) + list(INT_FORMATS)
FAULT_GROUPS = ["exp","sign","mant_msb","mant_lsb"]
FAULT_LABELS = {
    "exp":      "Group 1 – Exponent Bits",
    "sign":     "Group 2 – Sign Bit",
    "mant_msb": "Group 3 – Mantissa / Value MSB  (top 50 %)",
    "mant_lsb": "Group 4 – Mantissa / Value LSB  (bottom 50 %)",
}

# ── Bit helpers ───────────────────────────────────────────────────────────────
def float_bit_positions(E, M, group):
    if group == "sign": return [15]
    if group == "exp":  return list(range(M, M + E))
    n = math.ceil(M / 2)
    return list(range(M - n, M)) if group == "mant_msb" else list(range(0, M - n))

def int_bit_positions(bits, group):
    nv = bits - 1; n = math.ceil(nv / 2)
    if group == "sign": return [bits - 1]
    if group == "exp":  return []
    return list(range(bits - 1 - n, bits - 1)) if group == "mant_msb" else list(range(0, nv - n))

def make_xor_masks(bp, nf):
    if not bp: return np.zeros(nf, dtype=np.uint16)
    return np.array([np.uint16(1 << int(b)) for b in np.random.choice(bp, size=nf)], dtype=np.uint16)

# ── CUDA kernels ──────────────────────────────────────────────────────────────
_PU = r"""
__device__ __forceinline__ int ci(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}
extern "C" __global__ void pack_fp32(const float* __restrict__ s,unsigned short* __restrict__ d,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int E=E_BITS,M=M_BITS,bias=(1<<(E-1))-1,mx=(1<<E)-2;
    float w=s[i];unsigned short sg=(w<0.f)?1u:0u;float a=fabsf(w);
    if(a<0.5f*__expf((1-bias)*0.693147f)){d[i]=sg<<15;return;}
    float mv=(2.f-__expf(-M*0.693147f))*__expf((mx-bias)*0.693147f);a=fminf(a,mv);
    int eu=ci((int)floorf(log2f(a+1e-45f)),1-bias,mx-bias),eb=eu+bias;
    float fr=fminf(fmaxf(a/__expf(eu*0.693147f)-1.f,0.f),1.f-__expf(-M*0.693147f));
    unsigned int mn=(unsigned int)roundf(fr*(float)(1<<M));if(mn>=(unsigned)(1<<M))mn=(1<<M)-1;
    d[i]=(sg<<15)|((unsigned short)(eb&((1<<E)-1))<<M)|(unsigned short)(mn&((1<<M)-1));
}
extern "C" __global__ void unpack_fp32(const unsigned short* __restrict__ s,float* __restrict__ d,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int E=E_BITS,M=M_BITS,bias=(1<<(E-1))-1,mx=(1<<E)-2;
    unsigned short w=s[i];int sg=(w>>15)&1,eb=(w>>M)&((1<<E)-1),mi=w&((1<<M)-1);
    if(eb==0){d[i]=0.f;return;}if(eb==(1<<E)-1)eb=mx;
    d[i]=(sg?-1.f:1.f)*(1.f+(float)mi/(float)(1<<M))*__expf((eb-bias)*0.693147f);
}
"""
_INJ = r"""
extern "C" __global__ void inject_xor(unsigned short* __restrict__ p,const int* __restrict__ idx,
    const unsigned short* __restrict__ mk,int nf,int tot){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nf)return;
    int wi=idx[k];if(wi>=0&&wi<tot)p[wi]^=mk[k];
}
"""
_pu_cache: dict = {}; _inj_fn = None; _BLK = 256
def _run(fn,n,*a): fn(((n+_BLK-1)//_BLK,),(_BLK,),a)
def _get_pu(E,M):
    k=(E,M)
    if k not in _pu_cache:
        if not CUPY_OK: raise RuntimeError("CuPy unavailable.")
        mod=RawModule(code=_PU,options=(f"-DE_BITS={E}",f"-DM_BITS={M}"),name_expressions=["pack_fp32","unpack_fp32"])
        _pu_cache[k]={"pack":mod.get_function("pack_fp32"),"unpack":mod.get_function("unpack_fp32")}
    return _pu_cache[k]
def _get_inj():
    global _inj_fn
    if _inj_fn is None:
        if not CUPY_OK: raise RuntimeError("CuPy unavailable.")
        _inj_fn=RawModule(code=_INJ,options=(),name_expressions=["inject_xor"]).get_function("inject_xor")
    return _inj_fn

# ── NumPy fallbacks ───────────────────────────────────────────────────────────
def _np_pack(w,E,M):
    bias=(1<<(E-1))-1;meb=(1<<E)-2;mx=(2.0-2.0**-M)*2.0**(meb-bias)
    sg=(w<0).astype(np.uint16);a=np.clip(np.abs(w),0.0,mx).astype(np.float32)
    ftz=a<0.5*2.0**(1-bias);safe=np.where(a>0,a,np.float32(1.0))
    eu=np.clip(np.floor(np.log2(safe)).astype(int),1-bias,meb-bias)
    eb=(eu+bias).astype(np.uint16)
    frac=np.clip(a/np.ldexp(1.0,eu)-1.0,0.0,1.0-2.0**-M)
    mant=np.minimum(np.round(frac*(1<<M)).astype(np.uint16),np.uint16((1<<M)-1))
    wd=(sg<<15)|(eb<<M)|mant;wd[ftz]=(sg[ftz]<<15);return wd.astype(np.uint16)
def _np_unpack(p,E,M):
    bias=(1<<(E-1))-1;meb=(1<<E)-2
    s=((p>>15)&1).astype(np.float32);eb=((p>>M)&((1<<E)-1)).astype(int)
    mi=(p&((1<<M)-1)).astype(np.float32);eu=np.clip(eb,1,meb)-bias
    val=(1.0+mi/(1<<M))*np.ldexp(1.0,eu);val[eb==0]=0.0
    return np.where(s>0,-val,val).astype(np.float32)
def _quant_int(w,bits):
    mx=(1<<(bits-1))-1;mn=-(1<<(bits-1))
    ab=float(np.max(np.abs(w)));sc=ab/mx if ab>0 else 1.0
    wi=np.clip(np.round(w/sc),mn,mx).astype(np.int32)
    return (wi&((1<<bits)-1)).astype(np.uint16),sc
def _dequant_int(p,bits,sc):
    mask=(1<<bits)-1;sb=1<<(bits-1);wi=p.astype(np.int32)&mask
    return ((np.where(wi&sb,wi-(1<<bits),wi))*sc).astype(np.float32)

# ── Weight stores ─────────────────────────────────────────────────────────────
class FloatStore:
    def __init__(self,model,E,M):
        self.E,self.M=E,M;self.meta=[p.data.shape for p in model.parameters()]
        self.sizes=[p.numel() for p in model.parameters()]
        flat=np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()]).astype(np.float32)
        if CUPY_OK and DEVICE.type=="cuda":
            kk=_get_pu(E,M);src=cp.asarray(flat);n=int(src.size)
            self._gpu=cp.empty(n,dtype=cp.uint16);_run(kk["pack"],n,src,self._gpu,np.int32(n));self._cpu=None
        else:
            self._gpu=None;self._cpu=_np_pack(flat,E,M)
    def clone(self):
        c=object.__new__(FloatStore);c.E,c.M,c.meta,c.sizes=self.E,self.M,self.meta,self.sizes
        c._gpu=self._gpu.copy() if self._gpu is not None else None
        c._cpu=None if self._gpu is not None else self._cpu.copy();return c
    def inject(self,bp,nf):
        if not bp: return
        masks=make_xor_masks(bp,nf)
        if self._gpu is not None:
            tot=int(self._gpu.size)
            _run(_get_inj(),nf,self._gpu,cp.array(np.random.randint(0,tot,nf),dtype=cp.int32),
                 cp.array(masks,dtype=cp.uint16),np.int32(nf),np.int32(tot))
        else:
            for idx,mk in zip(np.random.randint(0,self._cpu.size,nf),masks): self._cpu[idx]^=mk
    def apply(self,model):
        if self._gpu is not None:
            kk=_get_pu(self.E,self.M);n=int(self._gpu.size)
            dst=cp.empty(n,dtype=cp.float32);_run(kk["unpack"],n,self._gpu,dst,np.int32(n))
            flat=torch.as_tensor(dst,device=DEVICE)
        else:
            flat=torch.from_numpy(_np_unpack(self._cpu,self.E,self.M)).to(DEVICE)
        off=0
        for p,sz,sh in zip(model.parameters(),self.sizes,self.meta):
            p.data=flat[off:off+sz].reshape(sh).float();off+=sz

class IntStore:
    def __init__(self,model,bits):
        self.bits=bits;self.meta=[p.data.shape for p in model.parameters()]
        self.sizes=[p.numel() for p in model.parameters()];self.scales=[];self._pkd=[]
        for p in model.parameters():
            pk,sc=_quant_int(p.data.cpu().numpy().ravel().astype(np.float32),bits)
            self._pkd.append(pk.copy());self.scales.append(sc)
        self._cumsz=np.concatenate([[0],np.cumsum(self.sizes)])
    def clone(self):
        c=object.__new__(IntStore);c.bits,c.meta,c.sizes=self.bits,self.meta,self.sizes
        c.scales=self.scales;c._pkd=[a.copy() for a in self._pkd];c._cumsz=self._cumsz;return c
    def inject(self,bp,nf):
        if not bp: return
        masks=make_xor_masks(bp,nf)
        for g,mk in zip(np.random.randint(0,int(self._cumsz[-1]),nf),masks):
            pi=int(np.searchsorted(self._cumsz[1:],g,side="right"))
            self._pkd[pi][int(g-self._cumsz[pi])]^=mk
    def apply(self,model):
        for p,pk,sc,sh in zip(model.parameters(),self._pkd,self.scales,self.meta):
            p.data=torch.from_numpy(_dequant_int(pk,self.bits,sc)).reshape(sh).to(DEVICE).float()

# ── Data & model ──────────────────────────────────────────────────────────────
def get_loader():
    """Single loader — full 10,000-sample test set for all evaluations."""
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    ds=datasets.CIFAR10("./data",train=False,download=True,transform=tf)
    return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

def load_model():
    """MobileNetV2 for CIFAR-10: stride-1 stem, 10-class head."""
    m=models.mobilenet_v2(weights=None)
    old=m.features[0][0]
    m.features[0][0]=nn.Conv2d(old.in_channels,old.out_channels,
                                old.kernel_size,stride=1,padding=old.padding,bias=False)
    m.classifier[1]=nn.Linear(m.last_channel,NUM_CLASSES)
    m.load_state_dict(torch.load(PTH_PATH,map_location=DEVICE))
    return m.to(DEVICE).eval()

# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model,store,loader):
    store.apply(model);model.float().eval();c=t=0
    for imgs,labels in loader:
        imgs=imgs.float().to(DEVICE,non_blocking=True)
        labels=labels.to(DEVICE,non_blocking=True)
        c+=model(imgs).argmax(1).eq(labels).sum().item();t+=labels.size(0)
    return 100.0*c/t

# ── Main experiment ───────────────────────────────────────────────────────────
def run():
    backend="CuPy CUDA" if (CUPY_OK and DEVICE.type=="cuda") else "NumPy CPU"
    print(f"Model     : {MODEL_NAME}  [stride-1 stem, {NUM_CLASSES}-class head]")
    print(f"Dataset   : {DATASET_NAME}  ({NUM_CLASSES} classes, 32×32)  |  Weights : {PTH_PATH}")
    print(f"Device    : {DEVICE}  [{backend}]")
    print(f"Eval set  : FULL test set (10,000 samples)")
    print(f"Skip rule : formats with PTQ drop > {ACCURACY_DROP_THRESHOLD}% shown in bar only")
    print("═"*72)

    loader=get_loader(); base=load_model()

    @torch.no_grad()
    def _plain(m,ld):
        m.float().eval();c=t=0
        for imgs,labels in ld:
            imgs,labels=imgs.float().to(DEVICE,non_blocking=True),labels.to(DEVICE,non_blocking=True)
            c+=m(imgs).argmax(1).eq(labels).sum().item();t+=labels.size(0)
        return 100.0*c/t

    fp32_acc=_plain(base,loader)
    skip_threshold=fp32_acc-ACCURACY_DROP_THRESHOLD
    print(f"Baseline FP32 : {fp32_acc:.2f}%")
    print(f"Skip if clean < {skip_threshold:.2f}%  (drop > {ACCURACY_DROP_THRESHOLD}%)\n")

    results={}; clean_acc={}; skipped=[]

    for fmt in ALL_FMTS:
        if fmt in FLOAT_FORMATS:
            E,M,_=FLOAT_FORMATS[fmt]
            if CUPY_OK and DEVICE.type=="cuda": _get_pu(E,M);_get_inj()
            store_clean=FloatStore(base,E,M)
            bpg={g:float_bit_positions(E,M,g) for g in FAULT_GROUPS}
        else:
            bits=INT_FORMATS[fmt];store_clean=IntStore(base,bits)
            bpg={g:int_bit_positions(bits,g) for g in FAULT_GROUPS}

        ca=evaluate(copy.deepcopy(base),store_clean,loader)
        clean_acc[fmt]=ca; drop_from_fp32=fp32_acc-ca

        # ── EARLY-SKIP CHECK ──────────────────────────────────────────────
        if drop_from_fp32>ACCURACY_DROP_THRESHOLD:
            skipped.append(fmt)
            print(f"[{fmt:>8s}]  clean={ca:.2f}%  (FP32 drop={drop_from_fp32:.2f}%)  "
                  f"*** SKIPPED — drop exceeds {ACCURACY_DROP_THRESHOLD}% threshold ***")
            continue

        # ── Fault injection — FULL 10k test set ───────────────────────────
        results[fmt]={g:{} for g in FAULT_GROUPS}
        print(f"[{fmt:>8s}]  clean={ca:.2f}%  (FP32 drop={drop_from_fp32:.2f}%)")

        for group in FAULT_GROUPS:
            bp=bpg[group]
            print(f"           {FAULT_LABELS[group]:45s}  valid_bits={len(bp):2d}  ",end="",flush=True)
            for nf in N_FAULTS:
                accs=[]
                for _ in range(N_TRIALS):
                    s=store_clean.clone();s.inject(bp,nf)
                    accs.append(evaluate(copy.deepcopy(base),s,loader))
                arr=np.array(accs)
                results[fmt][group][nf]={"mean":float(arr.mean()),"std":float(arr.std())}
            d1=ca-results[fmt][group][1]["mean"]; d8=ca-results[fmt][group][N_FAULTS[-1]]["mean"]
            print(f"drop@1={d1:.2f}%   drop@8={d8:.2f}%")
        print()

    if skipped:
        print("─"*72)
        print(f"SKIPPED formats (PTQ drop > {ACCURACY_DROP_THRESHOLD}% from FP32={fp32_acc:.2f}%):")
        for fmt in skipped:
            print(f"  {fmt:>8s}  clean={clean_acc[fmt]:.2f}%  drop={fp32_acc-clean_acc[fmt]:.2f}%")
        print("─"*72+"\n")

    return fp32_acc,clean_acc,results,skipped

# ── Plot ──────────────────────────────────────────────────────────────────────
_CA:dict={}; _RS:dict={}; _SK:list=[]

def _dd(ax,group,lbl,yl):
    for fmt in [f for f in ALL_FMTS if f not in _SK]:
        ca=_CA[fmt];ys=np.array([ca-_RS[fmt][group][nf]["mean"] for nf in N_FAULTS])
        err=np.array([_RS[fmt][group][nf]["std"] for nf in N_FAULTS])
        ax.plot(N_FAULTS,ys,color=FMT_COLOR[fmt],ls=FMT_LS[fmt],marker=FMT_MK[fmt],zorder=3,label=fmt)
        ax.fill_between(N_FAULTS,ys-err,ys+err,color=FMT_COLOR[fmt],alpha=0.10,linewidth=0)
    ax.set_title(f"{lbl}  {FAULT_LABELS[group]}",loc="left",fontweight="bold",fontsize=7)
    ax.set_xlabel("Number of Bit-Flips")
    if yl: ax.set_ylabel(r"Accuracy Drop  $\Delta$ (%)")
    ax.set_xticks(N_FAULTS);ax.set_axisbelow(True)

def plot(fp32_acc,clean_acc,results,skipped):
    global _CA,_RS,_SK; _CA,_RS,_SK=clean_acc,results,skipped
    fig=plt.figure(figsize=(14,10))
    fig.suptitle(
        f"PTQ WOQ — Fault Robustness  ({MODEL_NAME} / {DATASET_NAME}, Full 10k Test Set)\n"
        f"Hatched bars: PTQ accuracy drop > {ACCURACY_DROP_THRESHOLD}% — fault injection skipped",
        fontsize=9,fontweight="bold",y=1.01)
    gs=gridspec.GridSpec(3,2,figure=fig,height_ratios=[1.0,1.35,1.35],hspace=0.52,wspace=0.28)

    ax0=fig.add_subplot(gs[0,:]);vals=[clean_acc[f] for f in ALL_FMTS]
    for i,(fmt,v) in enumerate(zip(ALL_FMTS,vals)):
        is_sk=fmt in skipped
        ax0.bar(i,v,color=FMT_COLOR[fmt],edgecolor="black",linewidth=0.45,width=0.65,zorder=3,
                hatch="////" if is_sk else "",alpha=0.55 if is_sk else 1.0)
        ax0.text(i,v+0.12,f"{v:.1f}\n(skip)" if is_sk else f"{v:.1f}",
                 ha="center",va="bottom",fontsize=4.5,color="red" if is_sk else "black")
    ax0.axhline(fp32_acc,color="#CC0000",ls="--",lw=1.0,zorder=4,label=f"FP32 ({fp32_acc:.1f}%)")
    ax0.axhline(fp32_acc-ACCURACY_DROP_THRESHOLD,color="orange",ls=":",lw=1.0,zorder=4,
                label=f"Skip threshold ({fp32_acc-ACCURACY_DROP_THRESHOLD:.1f}%)")
    ax0.set_xticks(range(len(ALL_FMTS)));ax0.set_xticklabels(ALL_FMTS,rotation=32,ha="right")
    ax0.set_title(f"(a) Clean Accuracy After PTQ  [{MODEL_NAME} / {DATASET_NAME}]",
                  loc="left",fontweight="bold",fontsize=7)
    ax0.set_ylabel("Top-1 Accuracy (%)");ax0.set_ylim(max(0,min(vals)-6),min(100,max(vals)+3))
    ax0.legend(handlelength=1.2,borderpad=0.4,fontsize=6,loc="lower right")
    for xp in (6.5,9.5): ax0.axvline(xp,color="0.55",lw=0.7,ls=":")
    yl=min(vals)-5
    for xc,lb in [(3.0,"Custom Float-16"),(8.0,"Custom Float-8"),(11.5,"INT")]:
        ax0.text(xc,yl,lb,ha="center",fontsize=5.5,color="0.45",style="italic")
    ax0.set_axisbelow(True)

    _dd(fig.add_subplot(gs[1,0]),"exp","(b)",True)
    _dd(fig.add_subplot(gs[1,1]),"sign","(c)",False)
    _dd(fig.add_subplot(gs[2,0]),"mant_msb","(d)",True)
    _dd(fig.add_subplot(gs[2,1]),"mant_lsb","(e)",False)

    flines=[Line2D([],[],color="k",ls=s,lw=1.0,label=l) for s,l in
            [("-","─── Custom Float-16"),("--","--- Custom Float-8"),(":", "··· INT")]]
    skip_patch=Patch(facecolor="grey",hatch="////",alpha=0.5,edgecolor="black",
                     label=f"Skipped (drop>{ACCURACY_DROP_THRESHOLD}%)")
    fhandles=[Line2D([0],[0],color=FMT_COLOR[f],ls=FMT_LS[f],marker=FMT_MK[f],
                     lw=1.0,markersize=3.0,label=f) for f in ALL_FMTS if f not in skipped]
    fig.legend(handles=flines+[skip_patch]+fhandles,loc="lower center",ncol=7,
               bbox_to_anchor=(0.5,-0.12),fontsize=6.0,framealpha=0.9,edgecolor="0.75",
               title="Format  (hatched = skipped, no fault injection)",title_fontsize=6)
    plt.tight_layout(rect=[0,0.10,1,1])
    for ext in ("pdf","png"):
        fn=f"{OUT_PREFIX}.{ext}";plt.savefig(fn,**({"dpi":300} if ext=="png" else {}));print(f"Saved → {fn}")
    plt.show()

    active=[f for f in ALL_FMTS if f not in skipped]
    if active:
        nf8=N_FAULTS[-1]
        dm=np.array([[clean_acc[fmt]-results[fmt][grp][nf8]["mean"] for grp in FAULT_GROUPS] for fmt in active])
        fig2,ax2=plt.subplots(figsize=(6,max(4,len(active)*0.55)));vmax=max(dm.max(),1.0)
        im=ax2.imshow(dm,aspect="auto",cmap="Reds",vmin=0,vmax=vmax)
        plt.colorbar(im,ax=ax2,label=f"Mean accuracy drop at {nf8} faults (%)")
        ax2.set_xticks(range(4));ax2.set_xticklabels(["Exponent Bits","Sign Bit",
            "Mantissa / Value\nMSB (top 50%)","Mantissa / Value\nLSB (bottom 50%)"],fontsize=6.5,ha="center")
        ax2.set_yticks(range(len(active)));ax2.set_yticklabels(active,fontsize=6.5)
        ax2.set_title(f"Accuracy Drop Heatmap @ {nf8} Bit-Flips\n{MODEL_NAME} / {DATASET_NAME}  (skipped excluded)",
                      fontweight="bold",fontsize=8)
        thresh=vmax*0.55
        for i in range(len(active)):
            for j in range(4):
                v=dm[i,j];ax2.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=5.5,
                                    color="white" if v>thresh else "black")
        plt.tight_layout()
        for ext in ("pdf","png"):
            fn=f"{OUT_PREFIX}_heatmap.{ext}";plt.savefig(fn,**({"dpi":300} if ext=="png" else {}))
        plt.show();print(f"Saved → {OUT_PREFIX}_heatmap.pdf / .png")

    W=14;sep="═"*110;print(f"\n{sep}")
    print(f"  PTQ FAULT ROBUSTNESS SUMMARY  —  {MODEL_NAME} / {DATASET_NAME}")
    grp_names="  ".join(f"{FAULT_LABELS[g].split('–',1)[-1].strip()[:W]:>{W}}" for g in FAULT_GROUPS)
    print(f"  {'Format':<10} {'Clean%':>7}  {'FP32Δ':>6}  {'Status':<12}  {grp_names}")
    print(f"  {'':─<10} {'':─>7}  {'':─>6}  {'':─<12}  "+"  ".join("─"*W for _ in FAULT_GROUPS))
    for fmt in ALL_FMTS:
        ca=clean_acc[fmt]
        if fmt in skipped:
            print(f"  {fmt:<10} {ca:>6.2f}%  {fp32_acc-ca:>5.2f}%  {'SKIPPED':<12}  (fault injection not performed)")
        else:
            row=f"  {fmt:<10} {ca:>6.2f}%  {fp32_acc-ca:>5.2f}%  {'OK':<12}  "
            for g in FAULT_GROUPS:
                row+=(f"  {ca-results[fmt][g][1]['mean']:>4.1f}/{ca-results[fmt][g][4]['mean']:>4.1f}/{ca-results[fmt][g][8]['mean']:>5.1f}%  ")
            print(row)
    print(sep)

    out={"model":MODEL_NAME,"dataset":DATASET_NAME,"num_classes":NUM_CLASSES,
         "ptq_mode":"weight-only","eval_set":"full_test_10000","n_trials":N_TRIALS,
         "fp32_baseline":fp32_acc,"accuracy_drop_threshold":ACCURACY_DROP_THRESHOLD,
         "skipped_formats":skipped,
         "formats":{"float16":[k for k in FLOAT_FORMATS if FLOAT_FORMATS[k][2]==16],
                    "float8":[k for k in FLOAT_FORMATS if FLOAT_FORMATS[k][2]==8],"int":list(INT_FORMATS.keys())},
         "fault_groups":FAULT_LABELS,"clean_acc":clean_acc,
         "results":{fmt:{grp:{str(nf):results[fmt][grp][nf] for nf in N_FAULTS} for grp in FAULT_GROUPS} for fmt in results}}
    fn=f"{OUT_PREFIX}_results.json"
    with open(fn,"w") as f: json.dump(out,f,indent=2)
    print(f"Saved → {fn}")

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    if DEVICE.type!="cuda": print("[WARN] No CUDA GPU – NumPy fallback (slow).\n")
    print("="*72)
    print(f"  PTQ-WOQ  —  {MODEL_NAME} / {DATASET_NAME}  [Full 10k test set]")
    print(f"  Early-skip threshold : {ACCURACY_DROP_THRESHOLD}% drop from FP32")
    print("="*72+"\n")
    fp32_acc,clean_acc,results,skipped=run()
    plot(fp32_acc,clean_acc,results,skipped)