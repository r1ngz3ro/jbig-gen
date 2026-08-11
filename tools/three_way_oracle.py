#!/usr/bin/env python3
"""Three-way oracle: generator page model vs PDFium vs Ghostscript jbig2dec.

    python3 tools/three_way_oracle.py <count> <workdir>

Run from the repo root. Only files that BOTH decoders accept cleanly are
counted -- jbig2dec keeps writing a partially-composed page after a fatal
error, so comparing that against PDFium's clipped-and-continued output would
measure error-recovery policy rather than decode correctness.

Polarity: PDFium's harness emits the page inverted, and --dump-page is written
in that same polarity; Ghostscript jbig2dec emits true polarity. Each row's
padding bits are masked before comparing. See DECODERS.md.
"""
import subprocess,sys,os
GEN="./header"; PF="./jbig2dec/build/jbig2dec"; GS="alt/jbig2dec1/jbig2dec"
N=int(sys.argv[1]); D=sys.argv[2]; os.makedirs(D,exist_ok=True)
def rd(p):
    d=open(p,'rb').read()
    if not d.startswith(b'P4'): return None
    i=d.index(b'\n'); j=d.index(b'\n',i+1)
    try: w,h=map(int,d[i+1:j].split())
    except: return None
    return w,h,d[j+1:]
def norm(t,inv):
    w,h,b=t; stride=(w+7)//8; pad=stride*8-w; o=bytearray()
    if len(b)<stride*h: return None
    for r in range(h):
        row=bytearray(b[r*stride:(r+1)*stride])
        if inv: row=bytearray(x^0xFF for x in row)
        if pad: row[-1]&=(0xFF<<pad)&0xFF
        o+=row
    return bytes(o)
c={'all_agree':0,'pf_wrong':0,'gs_wrong':0,'model_wrong':0,'all_differ':0}
kept=0
for i in range(N):
    f=f"{D}/t.jb2"; m=f"{D}/m.pbm"; a=f"{D}/a.pbm"; b=f"{D}/b.pbm"
    for x in (f,m,a,b):
        if os.path.exists(x): os.remove(x)
    subprocess.run([GEN,f,"--dump-page",m],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if not (os.path.exists(f) and os.path.exists(m)): continue
    rp=subprocess.run([PF,f,a],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=25)
    rg=subprocess.run([GS,"-o",b,f],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=25)
    err=rg.stderr.decode("utf8","replace")
    if rp.returncode!=0 or rg.returncode!=0: continue
    if "FATAL" in err or "WARNING" in err: continue
    if not(os.path.exists(a) and os.path.exists(b)): continue
    tm,ta,tb=rd(m),rd(a),rd(b)
    if not(tm and ta and tb) or tm[:2]!=ta[:2] or tm[:2]!=tb[:2]: continue
    M=norm(tm,True); A=norm(ta,True); B=norm(tb,False)
    if None in (M,A,B): continue
    kept+=1
    if M==A==B: c['all_agree']+=1
    elif M==B and A!=B: c['pf_wrong']+=1
    elif M==A and B!=A: c['gs_wrong']+=1
    elif A==B and M!=A: c['model_wrong']+=1
    else: c['all_differ']+=1
    if M!=A or M!=B:
        n=sum(v for k,v in c.items() if k!='all_agree')
        if n<=4: os.replace(f,f"{D}/bad{n}.jb2")
print("comparable:",kept,c)
