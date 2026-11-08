"""Fig. 2 of the paper.  python fig_confirm.py <results/GW-HAC_v2> <out.pdf>"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("pdf"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7.5,"font.family":"serif","axes.linewidth":0.6,"pdf.fonttype":42})
R=sys.argv[1] if len(sys.argv)>1 else "results/GW-HAC_v2"
d=pd.read_csv(R.rstrip("/")+"/tables/confirm_per_dataset.csv")
u=(d[["GW-HAC","FL-U8","KM","GW-nw"]]<0.02).all(axis=1)
BLUE,GRAY,INK="#2a78d6","#8a8a84","#333333"
fig,axs=plt.subplots(1,2,figsize=(3.5,1.95),sharey=True)
lab={"1496_ringnorm":"ringnorm","1511_wholesale_customers":"wholesale","756_autoPrice":"autoPrice","40705_tokyo1":"tokyo1"}
off={"FL-U8":{"ringnorm":(0.08,0.13,"center"),"wholesale":(-0.12,0.13,"center"),"autoPrice":(0.03,0.0,"left"),"tokyo1":(0.03,-0.02,"left")}}
for ax,(c,name) in zip(axs,[("FL-U8","FL-U8"),("KM","$k$-means")]):
    x=np.linspace(-0.1,1,2)
    ax.fill_between(x,x-0.02,x+0.02,color="#e6e6e3",lw=0,zorder=0)
    ax.plot(x,x,color=GRAY,lw=0.6,zorder=1)
    ax.scatter(d.loc[u,c],d.loc[u,"GW-HAC"],s=9,facecolors="none",edgecolors=GRAY,linewidths=0.6,zorder=2,label=f"uninformative ({u.sum()})")
    ax.scatter(d.loc[~u,c],d.loc[~u,"GW-HAC"],s=9,color=BLUE,edgecolors="white",linewidths=0.3,zorder=3,label=f"informative ({(~u).sum()})")
    if c=="FL-U8":
        for k,v in lab.items():
            r=d[d.dataset==k].iloc[0]; dx,dy,ha=off[c][v]
            ax.annotate(v,(r[c],r["GW-HAC"]),(r[c]+dx,r["GW-HAC"]+dy),ha=ha,fontsize=6.5,color=INK,
                        arrowprops=dict(arrowstyle="-",lw=0.4,color=INK) if abs(dx)+abs(dy)>0.06 else None)
    ax.set_xlim(-0.08,1.0); ax.set_ylim(-0.08,1.0); ax.set_aspect("equal")
    ax.set_xlabel(f"ARI of {name}"); ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6,length=2.5,colors=INK)
axs[0].set_ylabel("ARI of GW-HAC")
axs[0].set_title("(a)",fontsize=7.5); axs[1].set_title("(b)",fontsize=7.5)
axs[1].legend(loc="upper left",frameon=False,fontsize=6.5,handletextpad=0.2,borderaxespad=0.1)
fig.tight_layout(pad=0.3,w_pad=0.8)
fig.savefig(sys.argv[2] if len(sys.argv)>2 else "confirm_scatter.pdf")
