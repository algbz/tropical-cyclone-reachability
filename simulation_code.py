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
T=12.0; DT=0.02; H=0.005; X0=np.array([0.055,-3.2,0.,0.]); GAMMA=.9; KAPPA=.35; PULSE_DURATION=.45; U_MAX=.9; RELIABILITY=.95

def lam(y): return .15+.85*np.exp(-(np.asarray(y)/1.25)**2)
def lam_prime(y):
    y=np.asarray(y); e=np.exp(-(y/1.25)**2); return .85*e*(-2*y/1.25**2)
def theta(y): return .5+.9*np.tanh(np.asarray(y)/1.8)
def rhs(S,U):
    x,y,z1,z2=S.T; th=theta(y); c,s=np.cos(th),np.sin(th); u1,u2=U.T
    return np.column_stack([lam(y)*x-KAPPA*x**3+.75*z1+.1*z2,.52+.06*x+.2*z2,-GAMMA*z1+c*u1-.25*s*u2,-GAMMA*z2+s*u1+.25*c*u2])
def rk2(S,U,h):
    k1=rhs(S,U); return S+h*rhs(S+.5*h*k1,U)
def baseline_trajectory(step=DT):
    n=int(round(T/step)); tr=np.empty((n+1,4)); S=X0[None,:].copy(); tr[0]=X0; z=np.zeros((1,2))
    for i in range(n): S=rk2(S,z,step); tr[i+1]=S[0]
    return tr
def _rk2_subset(S,U,h): return S if len(S)==0 or h<=0 else rk2(S,U,h)
def pulse_sweep(times,angles,duration=PULSE_DURATION,magnitude=U_MAX):
    times=np.asarray(times,float); angles=np.asarray(angles,float); n=int(round(T/DT)); starts=np.repeat(np.rint(times/DT).astype(int)*DT,len(angles)); ends=starts+duration; ang=np.tile(angles,len(times)); S=np.tile(X0,(len(starts),1)).astype(float); dirs=magnitude*np.column_stack([np.cos(ang),np.sin(ang)]); zero=np.zeros_like(dirs); tol=1e-13
    for i in range(n):
        t,tn=i*DT,(i+1)*DT; inactive=(tn<=starts+tol)|(t>=ends-tol); active=(t>=starts-tol)&(tn<=ends+tol); cross=(~inactive)&(~active)&(t<ends-tol)&(tn>ends+tol)
        if np.any(inactive): S[inactive]=_rk2_subset(S[inactive],zero[inactive],DT)
        if np.any(active): S[active]=_rk2_subset(S[active],dirs[active],DT)
        if np.any(cross):
            h1=ends[cross]-t
            for h in np.unique(np.round(h1,14)):
                ids=np.where(cross)[0][np.isclose(h1,h,atol=1e-13,rtol=0.)]; S[ids]=_rk2_subset(S[ids],dirs[ids],float(h)); S[ids]=_rk2_subset(S[ids],zero[ids],float(DT-h))
    return S.reshape(len(times),len(angles),4)
def jacobian(st):
    x,y,_,_=st; A=np.zeros((4,4)); A[0]=[lam(y)-3*KAPPA*x**2,lam_prime(y)*x,.75,.1]; A[1,0]=.06; A[1,3]=.2; A[2,2]=A[3,3]=-GAMMA; return A
def input_matrix(y):
    th=float(theta(y)); c,s=math.cos(th),math.sin(th); G=np.zeros((4,2)); G[2:]=[[c,-.25*s],[s,.25*c]]; return G
def terminal_sensitivity_and_kernel(tr):
    n=len(tr)-1; P=np.array([[1.,0.,0.,0.]]); sens=np.empty(n+1); lev=np.empty(n+1); ang=np.empty(n+1)
    def rec(i,P):
        sens[i]=np.linalg.norm(P); k=(P@input_matrix(tr[i,1]))[0]; lev[i]=np.linalg.norm(k); ang[i]=math.atan2(-k[1],-k[0])
    rec(n,P)
    for i in range(n-1,-1,-1): P=P@expm(jacobian(tr[i])*DT); rec(i,P)
    return sens,lev,np.unwrap(ang)
def refined_physical_boundary(n_angles=288):
    times=np.arange(3.8,4.401,DT); angles=np.linspace(0,2*np.pi,n_angles,endpoint=False); out=pulse_sweep(times,angles); m=-out[:,:,0].min(axis=1); j=np.where((m[:-1]>0)&(m[1:]<=0))[0][0]; a,b=times[j],times[j+1]; return float(a-m[j]*(b-a)/(m[j+1]-m[j]))
def exact_saddle_benchmark():
    horizon=8.; tt=np.linspace(0,horizon,401); ell,q0,b,umax=.5,.02,1.,.27; D=abs(q0)*np.exp(ell*horizon)*np.ones_like(tt); tau=np.minimum(PULSE_DURATION,np.maximum(horizon-tt,0)); R=abs(b)*umax/ell*np.exp(ell*(horizon-tt))*(1-np.exp(-ell*tau)); f=R>=D; return tt,R,D,tt[np.where(f)[0][-1]]
def _last_boundary_interpolation(tab):
    r=tab.sort_values("time"); vals=list(zip(r.time.to_numpy(),r.switch_probability.to_numpy())); i=[i for i in range(len(vals)-1) if vals[i][1]>=RELIABILITY and vals[i+1][1]<RELIABILITY][-1]; t1,p1=vals[i]; t2,p2=vals[i+1]; return float(t1+(RELIABILITY-p1)*(t2-t1)/(p2-p1))

def figure_1_and_A1():
    tr=baseline_trajectory(); grid=np.linspace(0,T,len(tr)); sens,lev,linang=terminal_sensitivity_and_kernel(tr); times=np.arange(0,10.51,.1); angles=np.linspace(0,2*np.pi,72,endpoint=False); sw=pulse_sweep(times,angles); mn=sw[:,:,0].min(1); mx=sw[:,:,0].max(1); best=np.unwrap(angles[sw[:,:,0].argmin(1)]); tp=refined_physical_boundary(); si=np.interp(times,grid,sens); li=np.interp(times,grid,lev)
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True); axs[0].fill_between(times,mn,mx,color=TEAL,alpha=.18,label="terminal x reachable range"); axs[0].plot(times,mn,lw=2.2,color=PURPLE,label="best terminal x"); axs[0].axhline(0,ls="--",lw=1.6,color=GRAY,label="branch boundary"); axs[0].axvline(tp,ls=":",lw=1.8,color=DARK,label="refined policy-class lock"); axs[0].set_ylabel("terminal branch coordinate"); axs[0].legend(frameon=True,ncol=2); axs[0].grid(alpha=.18); axs[1].plot(times,si/si.max(),lw=2.2,color=GREEN,label="branch-state sensitivity"); axs[1].plot(times,li/li.max(),lw=2.2,color=TEAL,label="branch input leverage"); axs[1].axvline(tp,ls=":",lw=1.8,color=DARK); axs[1].set_ylabel("normalized magnitude"); axs[1].legend(frameon=True); axs[1].grid(alpha=.18); axs[2].plot(times,best,lw=2.2,color=PURPLE,label="finite-pulse optimum"); lin=np.interp(times,grid,linang); off=2*np.pi*np.round((best[0]-lin[0])/(2*np.pi)); axs[2].plot(times,lin+off,"--",lw=2,color=GRAY,label="linear kernel prediction"); axs[2].axvline(tp,ls=":",lw=1.8,color=DARK); axs[2].set(xlabel="pulse start time",ylabel="optimal action angle [rad]"); axs[2].legend(frameon=True); axs[2].grid(alpha=.18); fig.suptitle("Deformation-steering normal form: sensitivity, interventionability, and action drift",y=.995); fig.tight_layout(); fig.savefig(FIG/"fig01.png",dpi=220); plt.close(fig)
    tt,R,D,lock=exact_saddle_benchmark(); fig,ax=plt.subplots(figsize=(8.2,4.8)); ax.plot(tt,R,lw=2.4,color=TEAL,label="maximum reachable shift R(t)"); ax.plot(tt,D,"--",lw=2.2,color=PURPLE,label="distance to basin boundary D"); ax.axvspan(lock,tt[-1],color=PURPLE,alpha=.12,label="locked"); ax.axvline(lock,ls=":",lw=1.6,color=GRAY); ax.set(xlabel="intervention start time",ylabel="terminal displacement",title="Exact local saddle benchmark"); ax.legend(frameon=True); ax.grid(alpha=.18); fig.tight_layout(); fig.savefig(FIG/"figA1.png",dpi=220); plt.close(fig)
def figure_2_information_boundary():
    c=pd.read_csv(DERIVED/"information_curve_highres.csv"); a=pd.read_csv(DERIVED/"information_boundary_highres.csv"); tp=4.15457130980865; ti=_last_boundary_interpolation(a); lat=pd.read_csv(DERIVED/"latent_uncertainty_curve.csv").latent_posterior_uncertainty.to_numpy(); fig,axs=plt.subplots(2,1,figsize=(8.8,6.9),sharex=True); axs[0].plot(c.time,c.switch_probability,marker="o",markersize=3,lw=1.25,color=PURPLE,label="resolved active-branch propagation")
    for k,r in a.iterrows(): axs[0].errorbar([r.time],[r.switch_probability],yerr=[[r.switch_probability-r.wilson_low],[r.wilson_high-r.switch_probability]],fmt="s",capsize=3,color=TEAL,label="independent high-sample audit" if k==a.index[0] else None)
    axs[0].axhline(RELIABILITY,ls="--",lw=1.2,color=GRAY,label="95% reliability"); axs[0].axvline(ti,ls="-.",lw=1.4,color=PURPLE,label="final information boundary"); axs[0].axvline(tp,ls=":",lw=1.6,color=DARK,label="physical policy boundary"); axs[0].axvspan(ti,tp,color=PURPLE,alpha=.1); [axs[0].axvline(t,lw=.7,color=GRAY,alpha=.25) for t in [3.5,3.75,4.0]]; axs[0].set_ylabel("switch probability"); axs[0].set_ylim(.55,1.01); axs[0].grid(alpha=.18); axs[0].legend(frameon=True,fontsize=8,ncol=2); axs[1].plot(c.time,lat,lw=1.8,color=TEAL,label="latent steering uncertainty"); axs[1].axvline(ti,ls="-.",lw=1.4,color=PURPLE); axs[1].axvline(tp,ls=":",lw=1.6,color=DARK); axs[1].axvspan(ti,tp,color=PURPLE,alpha=.1); [axs[1].axvline(t,lw=.7,color=GRAY,alpha=.25) for t in [3.5,3.75,4.0]]; axs[1].set_xlabel("intervention start time"); axs[1].set_ylabel("latent posterior uncertainty"); axs[1].grid(alpha=.18); axs[1].legend(frameon=True,fontsize=8); fig.tight_layout(); fig.savefig(FIG/"fig02.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_3_branch_split():
    s=pd.read_csv(DERIVED/"branch_split_exact_cadence.csv"); h=pd.read_csv(DERIVED/"branch_split_histogram.csv"); fig,axs=plt.subplots(1,3,figsize=(11.2,3.45),sharey=True)
    for ax,(_,r) in zip(axs,s.iterrows()):
        t=float(r.time); x=h.loc[np.isclose(h.time,t)]; ax.bar(x.bin_left,x.density,width=x.bin_right-x.bin_left,align="edge",color=PURPLE,alpha=.55,label="nonlinear posterior"); xx=np.linspace(-1.15,1.15,500); ax.plot(xx,norm.pdf(xx,loc=r.local_gaussian_mean,scale=r.local_gaussian_sd),"--",lw=1.8,color=TEAL,label="local Gaussian"); ax.axvline(0,ls=":",lw=1.3,color=GRAY); ax.set_title(f"t={t:.2f}: P={r.nonlinear_switch_probability:.2f}, Gaussian={r.local_gaussian_switch_probability:.2f}"); ax.set_xlabel("terminal branch coordinate"); ax.grid(alpha=.14)
    axs[0].set_ylabel("density"); axs[0].legend(frameon=True,fontsize=8); fig.suptitle("Finite uncertainty near the terminal branch boundary produces branch-split outcomes"); fig.tight_layout(); fig.savefig(FIG/"fig03.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_4_uncertainty_scaling():
    d=pd.read_csv(DERIVED/"small_uncertainty_gap_scaling_highres.csv"); g=d.groupby("epsilon").gap.agg(["mean","std"]).reset_index(); x=g.loc[g.epsilon<=.2,"epsilon"].to_numpy(); y=g.loc[g.epsilon<=.2,"mean"].to_numpy(); slope=float(np.dot(x,y)/np.dot(x,x)); pred=slope*x; r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)); th=.3756622281; fig,ax=plt.subplots(figsize=(7.4,4.8)); ax.errorbar(g.epsilon,g["mean"],yerr=g["std"],fmt="o",capsize=4,color=PURPLE,label="nonlinear repeated-seed audit"); xx=np.linspace(0,.31,180); ax.plot(xx,th*xx,ls="--",lw=1.8,color=TEAL,label=rf"local theory: {th:.3f}$\epsilon$"); ax.plot(xx,slope*xx,lw=1.8,color=GREEN,label=rf"fit for $\epsilon\leq0.20$: {slope:.3f}$\epsilon$"); ax.axvspan(.2,.31,color=GRAY,alpha=.06,label="outside fitted small-noise range"); ax.set_xlabel(r"posterior uncertainty amplitude $\epsilon$"); ax.set_ylabel(r"information-to-physical gap $\Delta t$"); ax.grid(alpha=.18); ax.legend(frameon=True,fontsize=8); ax.text(.02,.93,rf"$R^2={r2:.4f}$",transform=ax.transAxes,va="top"); fig.tight_layout(); fig.savefig(FIG/"fig04.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_5_robustness():
    s=pd.read_csv(DERIVED/"structural_robustness.csv"); d=pd.read_csv(DERIVED/"pulse_duration_robustness.csv"); m=pd.read_csv(DERIVED/"posterior_mean_offset_robustness.csv"); fig,axs=plt.subplots(1,3,figsize=(12.2,4)); p=s.pivot(index="deformation_multiplier",columns="coupling_multiplier",values="posterior_switch_probability"); im=axs[0].imshow(p.values,origin="lower",aspect="auto",vmin=min(.74,p.values.min()),vmax=.96,cmap=RECOVERY_CMAP); axs[0].set_xticks(range(len(p.columns)),[f"{x:.1f}" for x in p.columns]); axs[0].set_yticks(range(len(p.index)),[f"{x:.1f}" for x in p.index]); axs[0].set_xlabel("latent coupling multiplier"); axs[0].set_ylabel("deformation multiplier"); axs[0].set_title(r"(a) Reliability 0.20 before $t_{phys}$")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]): axs[0].text(j,i,f"{p.values[i,j]:.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04,label="best resolved switch probability"); axs[1].plot(d.duration,d.gap,marker="o",lw=1.8,color=PURPLE); axs[1].set_xlabel("pulse duration"); axs[1].set_ylabel(r"$t_{phys}-t_{info}$"); axs[1].set_title("(b) Policy-duration robustness"); axs[1].grid(alpha=.18); axs[2].plot(m.projected_mean_bias_sigma,m.switch_probability,marker="o",lw=1.8,color=TEAL); axs[2].axhline(RELIABILITY,ls="--",lw=1,color=GRAY); axs[2].set_xlabel(r"posterior-mean bias [$\sigma$]"); axs[2].set_ylabel("best resolved switch probability at t=4.00"); axs[2].set_title("(c) Mean-offset robustness"); axs[2].grid(alpha=.18); fig.tight_layout(); fig.savefig(FIG/"fig05.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_6_phase():
    d=pd.read_csv(DERIVED/"observation_phase_sweep.csv"); ph=d.observation_phase.to_numpy(); gap=d.gap.to_numpy(); fig,ax=plt.subplots(figsize=(7.2,4.5)); ax.plot(ph,gap,marker="o",lw=1.8,color=PURPLE,label="phase-shift audit"); ax.axhline(float(gap[0]),ls="--",lw=1.3,color=TEAL,label="phase 0"); ax.fill_between(ph,gap.min(),gap.max(),color=LIGHT_PURPLE,alpha=.35); ax.set_xlabel("observation-grid phase"); ax.set_ylabel(r"information-to-physical gap $\Delta t$"); ax.set_xlim(-.005,.23); ax.grid(alpha=.18); ax.legend(frameon=True,fontsize=8); ax.text(.02,.96,f"range: {gap.min():.2f}--{gap.max():.2f}",transform=ax.transAxes,va="top",color=DARK); fig.tight_layout(); fig.savefig(FIG/"fig06.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_7_sampling_null():
    n=pd.read_csv(DERIVED/"dorian_sampling_geometry_null.csv"); fig,ax=plt.subplots(figsize=(7.3,4.6)); ax.plot(n.strength_multiplier,n.false_saddle_fraction,marker="o",lw=1.8,color=PURPLE,label=r"$P(\widehat{\det J}<0)$"); ax.plot(n.strength_multiplier,n.observed_or_more_negative_fraction,marker="s",lw=1.8,color=TEAL,label=r"$P(\widehat{\det J}\leq\det J_{obs})$"); ax.axvline(1,ls=":",lw=1.3,color=GRAY,label="matched gradient strength"); ax.set_xlabel("positive-det null gradient strength / observed strength"); ax.set_ylabel("simulation fraction"); ax.set_title("Dorian sampling-geometry stress test"); ax.grid(alpha=.18); ax.legend(frameon=True,fontsize=8); fig.tight_layout(); fig.savefig(FIG/"fig07.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def figure_8_dorian():
    a=pd.read_csv(OBS/"dorian_giv_affine_steering.csv"); r=pd.read_csv(OBS/"dorian_steering_robustness_summary.csv"); g=pd.read_csv(OBS/"dorian_steering_robustness_grid.csv"); s=pd.read_csv(OBS/"dorian_giv_layermean_soundings.csv"); d=a.merge(r[["mission","negative_fraction"]],on="mission",how="left").sort_values("midtime"); x=np.arange(len(d)); labels=[m[4:8] for m in d.mission]; fig,axs=plt.subplots(2,2,figsize=(10.4,7.5)); axs[0,0].plot(x,d.track_speed_ms,marker="o",lw=1.7,color=PURPLE); axs[0,0].set_xticks(x,labels,rotation=45); axs[0,0].set_ylabel(r"translation speed (m s$^{-1}$)"); axs[0,0].set_title("(a) Dorian translation during surveillance"); axs[0,0].grid(alpha=.18); axs[0,1].plot(x,d.negative_fraction,marker="o",lw=1.7,color=TEAL); axs[0,1].axhline(.5,ls="--",lw=1,color=GRAY,alpha=.65); axs[0,1].set_xticks(x,labels,rotation=45); axs[0,1].set_ylabel(r"fraction with $\det J<0$"); axs[0,1].set_ylim(-.03,1.03); axs[0,1].set_title("(b) Layer/radius robustness"); axs[0,1].grid(alpha=.18); gg=g[(g.mission=="20190902N1")&(g.bottom==850)&(g.top==300)&g.outer.isin([700.,900.])]
    for outer,z in gg.groupby("outer"): z=z.sort_values("inner"); axs[1,0].plot(z.inner,z.detJ,marker="o",lw=1.7,color=PURPLE if int(outer)==700 else GREEN,label=f"outer {int(outer)} km")
    axs[1,0].axhline(0,ls="--",lw=1,color=GRAY); axs[1,0].set_xlabel("inner radius (km)"); axs[1,0].set_ylabel(r"$\det J$"); axs[1,0].set_title("(c) Sep 2 near-storm 850--300 hPa"); axs[1,0].grid(alpha=.18); axs[1,0].legend(frameon=True); ss=s[(s.mission=="20190902N1")&(s.r_km>=300)&(s.r_km<=900)].copy(); X=np.c_[np.ones(len(ss)),ss.x_km.to_numpy()/1000,ss.y_km.to_numpy()/1000]; cu=np.linalg.lstsq(X,ss.u850_300.to_numpy(),rcond=None)[0]; cv=np.linalg.lstsq(X,ss.v850_300.to_numpy(),rcond=None)[0]; lim=950; gx=np.linspace(-lim,lim,15); gy=np.linspace(-lim,lim,15); XX,YY=np.meshgrid(gx,gy); UU=cu[0]+cu[1]*(XX/1000)+cu[2]*(YY/1000); VV=cv[0]+cv[1]*(XX/1000)+cv[2]*(YY/1000); axs[1,1].streamplot(gx,gy,UU,VV,density=.8,linewidth=.8,arrowsize=.7,color=GRAY); axs[1,1].quiver(ss.x_km,ss.y_km,ss.u850_300,ss.v850_300,angles="xy",scale_units="xy",scale=.06,width=.004,color=TEAL,label="layer-mean winds"); axs[1,1].scatter([0],[0],marker="*",s=90,color=PURPLE,label="storm center"); axs[1,1].set_xlim(-lim,lim); axs[1,1].set_ylim(-lim,lim); axs[1,1].set_aspect("equal"); axs[1,1].set_xlabel("storm-relative x (km)"); axs[1,1].set_ylabel("storm-relative y (km)"); axs[1,1].set_title("(d) Sep 2 sampled affine field"); axs[1,1].legend(frameon=True,fontsize=7); fig.tight_layout(); fig.savefig(FIG/"fig08.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def _short_date(m):
    dt=datetime.strptime("".join(ch for ch in str(m) if ch.isdigit())[:8],"%Y%m%d"); return f"{dt.strftime('%b')} {dt.day}"
def figure_B1_positive_control():
    d=pd.read_csv(OBS/"dorian_steering_robustness_summary.csv").merge(pd.read_csv(OBS/"dorian_giv_affine_steering.csv")[["mission","track_speed_ms"]],on="mission",how="left"); j=pd.read_csv(OBS/"joaquin_steering_robustness_summary.csv").merge(pd.read_csv(OBS/"joaquin_steering_base_bootstrap.csv")[["mission","track_speed_ms"]],on="mission",how="left"); fig,ax=plt.subplots(figsize=(7.4,4.8)); ax.scatter(d.track_speed_ms,d.negative_fraction,s=55,color=PURPLE,label="Dorian 2019"); ax.scatter(j.track_speed_ms,j.negative_fraction,s=70,marker="^",color=TEAL,label="Joaquin 2015"); ax.axhline(.5,ls="--",lw=1,color=GRAY,alpha=.65)
    for _,r in d[d.mission.isin(["20190902N1","20190903N1"])].iterrows(): ax.annotate(_short_date(r.mission),(r.track_speed_ms,r.negative_fraction),xytext=(5,5),textcoords="offset points",fontsize=7.5)
    for _,r in j.iterrows(): ax.annotate(_short_date(r.mission),(r.track_speed_ms,r.negative_fraction),xytext=(5,-12),textcoords="offset points",fontsize=7.5)
    ax.set_xlabel(r"observed translation speed (m s$^{-1}$)"); ax.set_ylabel(r"fraction of reductions with $\det J<0$"); ax.set_ylim(-.03,1.04); ax.grid(alpha=.18); ax.legend(frameon=True); ax.set_title("Observed affine-geometry robustness: Dorian and Joaquin"); fig.tight_layout(); fig.savefig(FIG/"figB1.png",dpi=320,bbox_inches="tight"); plt.close(fig)
def required_inputs():
    return [DERIVED/"information_curve_highres.csv",DERIVED/"information_boundary_highres.csv",DERIVED/"branch_split_exact_cadence.csv",DERIVED/"branch_split_histogram.csv",DERIVED/"small_uncertainty_gap_scaling_highres.csv",DERIVED/"structural_robustness.csv",DERIVED/"pulse_duration_robustness.csv",DERIVED/"posterior_mean_offset_robustness.csv",DERIVED/"dorian_sampling_geometry_null.csv",DERIVED/"observation_phase_sweep.csv",DERIVED/"latent_uncertainty_curve.csv",OBS/"dorian_giv_affine_steering.csv",OBS/"dorian_steering_robustness_summary.csv",OBS/"dorian_steering_robustness_grid.csv",OBS/"dorian_giv_layermean_soundings.csv",OBS/"joaquin_steering_robustness_summary.csv",OBS/"joaquin_steering_base_bootstrap.csv"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--check",action="store_true"); a=p.parse_args(); miss=[x for x in required_inputs() if not x.exists()]
    if miss: raise SystemExit("Missing required input files:\n"+"\n".join(map(str,miss)))
    if a.check: print(f"All {len(required_inputs())} required input files are present."); return
    for label,f in [("Fig. 1 + A1",figure_1_and_A1),("Fig. 2",figure_2_information_boundary),("Fig. 3",figure_3_branch_split),("Fig. 4",figure_4_uncertainty_scaling),("Fig. 5",figure_5_robustness),("Fig. 6",figure_6_phase),("Fig. 7",figure_7_sampling_null),("Fig. 8",figure_8_dorian),("Fig. B1",figure_B1_positive_control)]: print(f"Generating {label} ...",flush=True); f()
    print(f"Done. Generated all 10 figure files in {FIG}")
if __name__=="__main__": main()
