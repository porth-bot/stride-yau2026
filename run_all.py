"""Final paper numbers. Every output goes into the paper."""
import numpy as np, time
CAD = 29.4/(60*24); PMIN, PMAX = 0.5, 180.0; OS = 3
C_DUTY = (13/24)/365**(1/3)

def gen(S, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, S, CAD); 
    for g in [80,170,260,350]:
        gs = g*S/365; t = t[~((t>=gs)&(t<gs+4.5))]
    return (t - t[0])[rng.random(len(t))<0.88]

def inj(t, P0, t00, d0, dep, seed):
    rng = np.random.default_rng(seed); y = rng.normal(0,0.001,len(t))
    ph = (t-t00)%P0; y[np.minimum(ph,P0-ph)<=d0/2] -= dep; y -= y.mean()
    return y, np.full(len(t),1.0/len(t))

def nf(S):
    f,n = 1/PMAX, 0
    while f < 1/PMIN: f += (C_DUTY*f**(-1/3))*f/(S*OS); n+=1
    return n

def M(t): return int(np.sum(np.floor(t/PMIN).astype(np.int64)+3))

def sweep(t,y,w,P,d):
    ph=t%P; ent=(ph-d)%P; ext=ph
    pos=np.concatenate([ent,ext]); ds=np.concatenate([w*y,-w*y]); dr=np.concatenate([w,-w])
    o=np.argsort(pos,kind='stable'); pos,ds,dr=pos[o],ds[o],dr[o]
    s0=(w[ph<d]*y[ph<d]).sum(); r0=w[ph<d].sum()
    s=s0+np.cumsum(ds); r=np.clip(r0+np.cumsum(dr),0,1)
    den=r*(1-r); sr=np.where(den>1e-15,s*s/np.maximum(den,1e-300),0.0)
    j=int(np.argmax(sr)); return float(sr[j]), float((pos[j]+1e-12+d/2)%P)

def cands(t,d,pm,px):
    m=np.rint(t/CAD).astype(np.int64); g=np.unique((m[None,:]-m[:,None]).ravel())
    g=g[g>0].astype(float)*CAD; K=int(np.floor((t[-1]-t[0])/pm))+1; out=[]
    for dk in range(1,K+1):
        for c in (-1.,0.,1.): P=(g+c*d)/dk; out.append(P[(P>=pm)&(P<=px)])
    return np.unique(np.concatenate(out+[np.array([pm,px])])), len(g), K

# ── Paper numbers ──
print("TEST 1: Duality (500, full range)")
t1=gen(365); N1=len(t1); rng=np.random.default_rng(123); mm=0
for _ in range(500):
    P=rng.uniform(0.5,180); t0=rng.uniform(0,P); d=rng.uniform(0.02,0.3)
    if d>=P: continue
    pr=set(np.where(np.minimum((t1-t0)%P,P-(t1-t0)%P)<=d/2)[0]); du=set()
    for i,ti in enumerate(t1):
        for k in range(int(np.floor((ti-t0-d/2)/P))-1, int(np.ceil((ti-t0+d/2)/P))+2):
            if abs(ti-t0-k*P)<=d/2: du.add(i); break
    if pr!=du: mm+=1
print(f"  N={N1}, mismatches={mm}/500")

print("\nTEST 2: Pencil (centerline intercepts)")
obs=42; tobs=t1[obs]; print(f"  obs {obs}: t={tobs:.6f}")
ks=[-2,-1,0,1,5,20]; P1,P2=3.7,11.2
for k in ks:
    t0_a=tobs-k*P1; t0_b=tobs-k*P2
    slope=(t0_b-t0_a)/(P2-P1); intercept=t0_a-slope*P1
    assert abs(slope-(-k))<1e-12 and abs(intercept-tobs)<1e-12
    print(f"    k={k:3d}: slope={slope:+.1f}, intercept={intercept:.6f}")
print(f"  all intercept (0, {tobs:.6f}): PASS")

print("\nTEST 3: Redundancy table")
for S in (365,730,1460):
    t=gen(S); N=len(t); m=M(t); n=nf(S); print(f"  {S}d: N={N}, M={m}, Nf={n}, ratio={n*N/m:.1f}")
cf=6*OS*PMIN/C_DUTY*((1/PMIN)**(1/3)-(1/PMAX)**(1/3)); print(f"  closed-form={cf:.1f}")

print("\nTEST 4: Depth")
P,t0,d=10.3271,3.11,0.12; y1,w1=inj(t1,10.5,2.3,0.15,0.005,42)
cnt=int((np.minimum((t1-t0)%P,P-(t1-t0)%P)<=d/2).sum())
dep=0
for i,ti in enumerate(t1):
    for k in range(int(np.floor((ti-t0)/P))-2, int(np.ceil((ti-t0)/P))+3):
        if abs(ti-t0-k*P)<=d/2: dep+=1; break
print(f"  primal={cnt}, dual={dep}, match={cnt==dep}")

print("\nTEST 5: Recovery (S=60d, P0=5.3, gap 25-27d)")
t5=gen(60,seed=77); t5=t5[~((t5>=25)&(t5<27))]; y5,w5=inj(t5,5.3,2.3,0.15,0.005,77)
N5=len(t5); ta=time.time()
C5,ng5,Km5=cands(t5,0.15,1.5,30.0); mids=np.concatenate([(C5[:-1]+C5[1:])/2,[1.5,30.0]])
best=(-1,0,0)
for P in mids:
    sr,t0=sweep(t5,y5,w5,P,0.15)
    if sr>best[0]: best=(sr,P,t0)
ts=time.time()-ta; bound5=3*ng5*Km5
print(f"  N={N5}, |C|={len(C5):,} (bound={bound5:,})")
print(f"  P*={best[1]:.5f}, t0*={best[2]:.4f}, |dP|={abs(best[1]-5.3):.4f}d")
print(f"  wall-clock={ts:.1f}s")
rng=np.random.default_rng(4); worst=-1
for _ in range(60000):
    P=rng.uniform(1.5,30); sr,_=sweep(t5,y5,w5,P,0.15)
    if sr>worst: worst=sr
print(f"  60k probes: max={worst:.4e}, beats={worst>best[0]*(1+1e-9)}")
rng2=np.random.default_rng(5); bad=0; tested=0
for j in rng2.integers(0,len(C5)-1,300):
    a,b=C5[j],C5[j+1]
    if b-a<1e-12: continue
    vals=[sweep(t5,y5,w5,a+u*(b-a),0.15)[0] for u in (0.12,0.37,0.61,0.88)]
    tested+=1
    if max(vals)-min(vals)>1e-9*max(1,max(vals)): bad+=1
print(f"  interval constancy: {tested-bad}/{tested}")

print("\nTEST 6: Sweep self-test")
rng=np.random.default_rng(0); ok=0
for _ in range(300):
    P=rng.uniform(2,14); sr,t0r=sweep(t5,y5,w5,P,0.15)
    ph=(t5-t0r)%P; m=np.minimum(ph,P-ph)<=0.15/2
    s=(w5[m]*y5[m]).sum(); r=w5[m].sum(); den=r*(1-r)
    d2=float(s*s/den) if den>1e-15 else 0.0
    if abs(sr-d2)<1e-12*max(1,sr): ok+=1
print(f"  {ok}/300")

print("\nTEST 7: Full-scale cost")
tf=gen(365); yf,wf=inj(tf,10.5,2.3,0.15,0.005,42)
reps=20; ta=time.time()
for P in np.random.default_rng(1).uniform(0.5,180,reps): sweep(tf,yf,wf,P,0.15)
pc=(time.time()-ta)/reps; Cb=3*int(365/CAD)*(int(365/PMIN)+1); eh=pc*Cb/3600
nfv=nf(365); gt=pc*nfv
print(f"  N={len(tf)}, per-cand={pc*1e3:.1f}ms, |C|bound={Cb:.2e}")
print(f"  STRIDE est: {eh:.0f}h, grid est: {gt:.1f}s, ratio: ~{eh*3600/gt:.0f}x")

print("\nTEST 8: Astropy wall-clock comparison")
try:
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u
    bls=BoxLeastSquares(tf*u.day, yf)
    f,ofir_periods=1/PMAX,[]
    while f<1/PMIN: ofir_periods.append(1/f); f+=(C_DUTY*f**(-1/3))*f/(365*OS)
    pgrid=np.sort(np.array(ofir_periods))*u.day
    ta=time.time(); _=bls.power(pgrid, 0.15*u.day); tb=time.time()-ta
    stride_est_full=eh*3600
    ratio_wc=stride_est_full/tb
    print(f"  astropy BLS: {len(pgrid)} periods in {tb:.1f}s")
    print(f"  STRIDE est:  {eh:.0f}h = {stride_est_full:.0f}s")
    print(f"  practical slowdown: ~{ratio_wc:.0f}x")
except ImportError:
    print("  skipped (astropy not installed)")
