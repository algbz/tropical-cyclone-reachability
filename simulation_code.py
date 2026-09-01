#!/usr/bin/env python3
"""Reproduce all figures for the reduced tropical-cyclone steering study."""
from __future__ import annotations
import argparse, math
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
DERIVED=ROOT/"derived"; OBS=ROOT/"observational"; FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True)
PURPLE="#6f4aa2"; TEAL="#2a9d8f"; GREEN="#3a8f5d"; GRAY="#7a7f85"; DARK="#222222"; LIGHT_PURPLE="#f2eef8"
RECOVERY_CMAP=LinearSegmentedColormap.from_list("recoverability",[LIGHT_PURPLE,"#bba7d1",PURPLE])
T=12.0; DT=0.02; X0=np.array([0.055,-3.2,0.,0.]); GAMMA=.9; KAPPA=.35; PULSE_DURATION=.45; U_MAX=.9; RELIABILITY=.95

def lam(y): return .15+.85*np.exp(-(np.asarray(y)/1.25)**2)
def lam_prime(y): y=np.asarray(y); return .85*np.exp(-(y/1.25)**2)*(-2*y/1.25**2)
def theta(y): return .5+.9*np.tanh(np.asarray(y)/1.8)
def rhs(S,U):
    x,y,z1,z2=S.T; th=theta(y); c,s=np.cos(th),np.sin(th); u1,u2=U.T
    return np.column_stack([lam(y)*x-KAPPA*x**3+.75*z1+.1*z2,.52+.06*x+.2*z2,-GAMMA*z1+c*u1-.25*s*u2,-GAMMA*z2+s*u1+.25*c*u2])
def rk2(S,U,h): k=rhs(S,U); return S+h*rhs(S+.5*h*k,U)
def baseline_trajectory(step=DT):
    n=int(round(T/step)); tr=np.empty((n+1,4)); S=X0[None,:].copy(); tr[0]=X0; z=np.zeros((1,2))
    for i in range(n): S=rk2(S,z,step); tr[i+1]=S[0]
    return tr
def pulse_sweep(times,angles,duration=PULSE_DURATION,magnitude=U_MAX):
    times=np.asarray(times,float); angles=np.asarray(angles,float); n=int(round(T/DT)); starts=np.repeat(np.rint(times/DT).astype(int)*DT,len(angles)); ends=starts+duration; ang=np.tile(angles,len(times)); S=np.tile(X0,(len(starts),1)).astype(float); dirs=magnitude*np.column_stack([np.cos(ang),np.sin(ang)]); zero=np.zeros_like(dirs); tol=1e-13
    for i in range(n):
        t,tn=i*DT,(i+1)*DT; inactive=(tn<=starts+tol)|(t>=ends-tol); active=(t>=starts-tol)&(tn<=ends+tol); cross=(~inactive)&(~active)&(t<ends-tol)&(tn>ends+tol)
        if np.any(inactive): S[inactive]=rk2(S[inactive],zero[inactive],DT)
        if np.any(active): S[active]=rk2(S[active],dirs[active],DT)
        if np.any(cross):
            h1=ends[cross]-t
            for h in np.unique(np.round(h1,14)):
                ids=np.where(cross)[0][np.isclose(h1,h,atol=1e-13,rtol=0.)]; S[ids]=rk2(S[ids],dirs[ids],float(h)); S[ids]=rk2(S[ids],zero[ids],float(DT-h))
    return S.reshape(len(times),len(angles),4)
def jacobian(st):
    x,y,_,_=st; A=np.zeros((4,4)); A[0]=[lam(y)-3*KAPPA*x*x,lam_prime(y)*x,.75,.1]; A[1,0]=.06; A[1,3]=.2; A[2,2]=A[3,3]=-GAMMA; return A
def input_matrix(y):
    th=float(theta(y)); c,s=math.cos(th),math.sin(th); G=np.zeros((4,2)); G[2:]=[[c,-.25*s],[s,.25*c]]; return G
def terminal_sensitivity_and_kernel(tr):
    n=len(tr)-1; P=np.array([[1.,0.,0.,0.]]); sens=np.empty(n+1); lev=np.empty(n+1); ang=np.empty(n+1)
    for i in range(n,-1,-1):
        sens[i]=np.linalg.norm(P); k=(P@input_matrix(tr[i,1]))[0]; lev[i]=np.linalg.norm(k); ang[i]=math.atan2(-k[1],-k[0]);
        if i: P=P@expm(jacobian(tr[i-1])*DT)
    return sens,lev,np.unwrap(ang)
def refined_physical_boundary(n_angles=288):
    t=np.arange(3.8,4.401,DT); o=pulse_sweep(t,np.linspace(0,2*np.pi,n_angles,endpoint=False)); m=-o[:,:,0].min(1); j=np.where((m[:-1]>0)&(m[1:]<=0))[0][0]; return float(t[j]-m[j]*(t[j+1]-t[j])/(m[j+1]-m[j]))
def _tinfo(a):
    v=list(zip(a.time,a.switch_probability)); i=[i for i in range(len(v)-1) if v[i][1]>=.95 and v[i+1][1]<.95][-1]; t1,p1=v[i]; t2,p2=v[i+1]; return t1+(.95-p1)*(t2-t1)/(p2-p1)
def figure_1_and_A1():
    tr=baseline_trajectory(); grid=np.linspace(0,T,len(tr)); s,l,a=terminal_sensitivity_and_kernel(tr); times=np.arange(0,10.51,.1); an=np.linspace(0,2*np.pi,72,endpoint=False); sw=pulse_sweep(times,an); mn=sw[:,:,0].min(1); mx=sw[:,:,0].max(1); best=np.unwrap(an[sw[:,:,0].argmin(1)]); tp=refined_physical_boundary(); fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True); axs[0].fill_between(times,mn,mx,color=TEAL,alpha=.18); axs[0].plot(times,mn,color=PURPLE); axs[0].axhline(0,ls="--",color=GRAY); axs[0].axvline(tp,ls=":",color=DARK); axs[1].plot(times,np.interp(times,grid,s)/s.max(),color=GREEN); axs[1].plot(times,np.interp(times,grid,l)/l.max(),color=TEAL); axs[2].plot(times,best,color=PURPLE); lin=np.interp(times,grid,a); axs[2].plot(times,lin+2*np.pi*np.round((best[0]-lin[0])/(2*np.pi)),"--",color=GRAY); [ax.grid(alpha=.18) for ax in axs]; axs[2].set_xlabel("pulse start time"); fig.tight_layout(); fig.savefig(FIG/"fig01.png",dpi=220); plt.close(fig); tt=np.linspace(0,8,401); D=.02*np.exp(.5*8)*np.ones_like(tt); tau=np.minimum(.45,np.maximum(8-tt,0)); R=.27/.5*np.exp(.5*(8-tt))*(1-np.exp(-.5*tau)); lock=tt[np.where(R>=D)[0][-1]]; fig,ax=plt.subplots(); ax.plot(tt,R,color=TEAL); ax.plot(tt,D,"--",color=PURPLE); ax.axvspan(lock,8,color=PURPLE,alpha=.12); fig.tight_layout(); fig.savefig(FIG/"figA1.png",dpi=220); plt.close(fig)
def figure_2_information_boundary():
    c=pd.read_csv(DERIVED/"information_curve_highres.csv"); a=pd.read_csv(DERIVED/"information_boundary_highres.csv"); lat=pd.read_csv(DERIVED/"latent_uncertainty_curve.csv"); ti=_tinfo(a); tp=4.15457130980865; fig,axs=plt.subplots(2,1,figsize=(8.8,6.9),sharex=True); axs[0].plot(c.time,c.switch_probability,"o-",color=PURPLE,ms=3); axs[0].axhline(.95,ls="--",color=GRAY); axs[0].axvline(ti,ls="-.",color=PURPLE); axs[0].axvline(tp,ls=":",color=DARK); axs[0].axvspan(ti,tp,color=PURPLE,alpha=.1); axs[1].plot(c.time,lat.latent_posterior_uncertainty,color=TEAL); axs[1].axvline(ti,ls="-.",color=PURPLE); axs[1].axvline(tp,ls=":",color=DARK); axs[1].axvspan(ti,tp,color=PURPLE,alpha=.1); axs[1].set_xlabel("intervention start time"); fig.tight_layout(); fig.savefig(FIG/"fig02.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_3_branch_split():
    s=pd.read_csv(DERIVED/"branch_split_exact_cadence.csv"); h=pd.read_csv(DERIVED/"branch_split_histogram.csv"); fig,axs=plt.subplots(1,3,figsize=(11.2,3.45),sharey=True)
    for ax,(_,r) in zip(axs,s.iterrows()):
        x=h[np.isclose(h.time,r.time)]; ax.bar(x.bin_left,x.density,width=x.bin_right-x.bin_left,align="edge",color=PURPLE,alpha=.55); xx=np.linspace(-1.15,1.15,500); ax.plot(xx,norm.pdf(xx,r.local_gaussian_mean,r.local_gaussian_sd),"--",color=TEAL); ax.axvline(0,ls=":",color=GRAY); ax.set_title(f"t={r.time:.2f}: P={r.nonlinear_switch_probability:.2f}")
    fig.tight_layout(); fig.savefig(FIG/"fig03.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_4_uncertainty_scaling():
    d=pd.read_csv(DERIVED/"small_uncertainty_gap_scaling_highres.csv"); g=d.groupby("epsilon").gap.agg(["mean","std"]).reset_index(); x=g[g.epsilon<=.2].epsilon.to_numpy(); y=g[g.epsilon<=.2]["mean"].to_numpy(); slope=float(np.dot(x,y)/np.dot(x,x)); xx=np.linspace(0,.31,180); fig,ax=plt.subplots(); ax.errorbar(g.epsilon,g["mean"],yerr=g["std"],fmt="o",color=PURPLE); ax.plot(xx,.3756622281*xx,"--",color=TEAL); ax.plot(xx,slope*xx,color=GREEN); fig.tight_layout(); fig.savefig(FIG/"fig04.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_5_robustness():
    s=pd.read_csv(DERIVED/"structural_robustness.csv"); d=pd.read_csv(DERIVED/"pulse_duration_robustness.csv"); m=pd.read_csv(DERIVED/"posterior_mean_offset_robustness.csv"); fig,axs=plt.subplots(1,3,figsize=(12.2,4)); p=s.pivot(index="deformation_multiplier",columns="coupling_multiplier",values="posterior_switch_probability"); axs[0].imshow(p.values,origin="lower",aspect="auto",cmap=RECOVERY_CMAP); axs[1].plot(d.duration,d.gap,"o-",color=PURPLE); axs[2].plot(m.projected_mean_bias_sigma,m.switch_probability,"o-",color=TEAL); axs[2].axhline(.95,ls="--",color=GRAY); fig.tight_layout(); fig.savefig(FIG/"fig05.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_6_phase():
    d=pd.read_csv(DERIVED/"observation_phase_sweep.csv"); fig,ax=plt.subplots(); ax.plot(d.observation_phase,d.gap,"o-",color=PURPLE); ax.axhline(d.gap.iloc[0],ls="--",color=TEAL); fig.tight_layout(); fig.savefig(FIG/"fig06.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_7_sampling_null():
    d=pd.read_csv(DERIVED/"dorian_sampling_geometry_null.csv"); fig,ax=plt.subplots(); ax.plot(d.strength_multiplier,d.false_saddle_fraction,"o-",color=PURPLE); ax.plot(d.strength_multiplier,d.observed_or_more_negative_fraction,"s-",color=TEAL); fig.tight_layout(); fig.savefig(FIG/"fig07.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_8_dorian():
    a=pd.read_csv(OBS/"dorian_giv_affine_steering.csv"); r=pd.read_csv(OBS/"dorian_steering_robustness_summary.csv"); d=a.merge(r,on="mission"); fig,axs=plt.subplots(1,2,figsize=(9,4)); axs[0].plot(np.arange(len(d)),d.track_speed_ms,"o-",color=PURPLE); axs[1].plot(np.arange(len(d)),d.negative_fraction,"o-",color=TEAL); fig.tight_layout(); fig.savefig(FIG/"fig08.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def _date(m):
    dt=datetime.strptime("".join(x for x in str(m) if x.isdigit())[:8],"%Y%m%d"); return f"{dt.strftime('%b')} {dt.day}"
def figure_B1_positive_control():
    d=pd.read_csv(OBS/"dorian_steering_robustness_summary.csv").merge(pd.read_csv(OBS/"dorian_giv_affine_steering.csv"),on="mission"); j=pd.read_csv(OBS/"joaquin_steering_robustness_summary.csv").merge(pd.read_csv(OBS/"joaquin_steering_base_bootstrap.csv"),on="mission"); fig,ax=plt.subplots(); ax.scatter(d.track_speed_ms,d.negative_fraction,color=PURPLE); ax.scatter(j.track_speed_ms,j.negative_fraction,color=TEAL,marker="^"); [ax.annotate(_date(r.mission),(r.track_speed_ms,r.negative_fraction),fontsize=7) for _,r in j.iterrows()]; fig.tight_layout(); fig.savefig(FIG/"figB1.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def required_inputs(): return [DERIVED/x for x in ["information_curve_highres.csv","information_boundary_highres.csv","branch_split_exact_cadence.csv","branch_split_histogram.csv","small_uncertainty_gap_scaling_highres.csv","structural_robustness.csv","pulse_duration_robustness.csv","posterior_mean_offset_robustness.csv","dorian_sampling_geometry_null.csv","observation_phase_sweep.csv","latent_uncertainty_curve.csv"]]+[OBS/x for x in ["dorian_giv_affine_steering.csv","dorian_steering_robustness_summary.csv","dorian_steering_robustness_grid.csv","dorian_giv_layermean_soundings.csv","joaquin_steering_robustness_summary.csv","joaquin_steering_base_bootstrap.csv"]]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--check",action="store_true"); a=p.parse_args(); missing=[x for x in required_inputs() if not x.exists()]
    if missing: raise SystemExit("Missing:\n"+"\n".join(map(str,missing)))
    if a.check: print("All required inputs are present."); return
    for f in [figure_1_and_A1,figure_2_information_boundary,figure_3_branch_split,figure_4_uncertainty_scaling,figure_5_robustness,figure_6_phase,figure_7_sampling_null,figure_8_dorian,figure_B1_positive_control]: f()
    print("Done. Generated fig01..fig08, figA1, figB1.")
if __name__=="__main__": main()
