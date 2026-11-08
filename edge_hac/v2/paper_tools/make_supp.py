"""Builds supplement.tex (numbering follows the elsarticle manuscript: Section 7, Table 3, Proposition 6, Theorem 4, Corollary 2) from the result folder.  python make_supp.py <results/GW-HAC_v2> <out/supplement.tex>"""
import pandas as pd, numpy as np, glob, json, re, ast, sys, os
R=(sys.argv[1] if len(sys.argv)>1 else "../../results/GW-HAC_v2").rstrip("/")+"/"
OUT=sys.argv[2] if len(sys.argv)>2 else "../../out/supplement.tex"
def esc(s): return str(s).replace("_","\\_").replace("&","\\&").replace("%","\\%")
def f3(x): return "--" if pd.isna(x) else f"{x+0.0:.3f}".replace("-0.000","0.000")
# ---- N,F,K
nfk={}
for lg in ("60_dev","60_val11","73_confirm"):
    for line in open(R+f"logs/{lg}.log"):
        m=re.match(r"\[done\] (\S+)(?: \((\w+)\))? N=(\d+) F=(\d+) K=(\d+)",line)
        if m: nfk[m.group(1)]=(int(m.group(3)),int(m.group(4)),int(m.group(5)))
fetch={}
for line in open(R+"logs/72_fetch.log"):
    m=re.match(r"(\d+_\S+) (\{.*\})$",line.strip())
    if m: fetch[m.group(1)]=ast.literal_eval(m.group(2))
# ---- dev / val11 ARI
fac=pd.concat(pd.read_csv(f) for f in glob.glob(R+"raw/v2fac_*.csv"))
sel={"GW-HAC":("HA-grid|native|interval",3),"FL-U8":("FL-U8",0),"KM":("KM",0)}
def ari_tab(group):
    rows=[]
    d=fac[fac.group==group]
    for ds,g in d.groupby("dataset"):
        r=dict(dataset=ds,N=nfk[ds][0],d=nfk[ds][1],K=nfk[ds][2])
        for k,(mth,l) in sel.items(): r[k]=g[(g.method==mth)&(g.level==l)].ari.mean()
        r["splits"]=g.split.nunique(); rows.append(r)
    return pd.DataFrame(rows).sort_values("dataset")
def small_table(df,cap,lab):
    s=[r"\begin{table}[h]\centering\small",f"\\caption{{{cap}}}\\label{{{lab}}}",r"\begin{tabular}{lrrrrrrr}\toprule",
       r"Dataset & $N$ & $d$ & $K$ & splits & GW-HAC & FL-U8 & $k$-means\\\midrule"]
    for _,r in df.iterrows():
        s.append(f"{esc(r.dataset)} & {r.N} & {r.d} & {r.K} & {r.splits} & {f3(r['GW-HAC'])} & {f3(r['FL-U8'])} & {f3(r.KM)}\\\\")
    s+= [r"\bottomrule\end{tabular}\end{table}"]; return "\n".join(s)
dev=ari_tab("dev"); val=ari_tab("val11")
# ---- confirmatory
c=pd.read_csv(R+"tables/confirm_per_dataset.csv")
u=(c[["GW-HAC","FL-U8","KM","GW-nw"]]<0.02).all(axis=1)
L=[r"\begin{longtable}{rlrrrrrrrc}",r"\caption{Confirmatory datasets: mean held-out ARI over splits. U: uninformative.}\label{tab:s-confirm}\\",
   r"\toprule OpenML id & Name & $N$ & $d$ & $K$ & GW-HAC & FL-U8 & $k$-means & GW-nw & U\\\midrule\endfirsthead",
   r"\toprule OpenML id & Name & $N$ & $d$ & $K$ & GW-HAC & FL-U8 & $k$-means & GW-nw & U\\\midrule\endhead",r"\bottomrule\endfoot"]
c["id"]=c.dataset.str.split("_").str[0].astype(int)
for i,r in c.sort_values("id").iterrows():
    fm=fetch.get(r.dataset,{}); nm=fm.get("openml_name",r.dataset.split("_",1)[1])
    N,dd,K=(fm["N"],fm["F"],fm["K"]) if fm else nfk[r.dataset]
    if len(nm)>28: nm=nm[:26]+"..."
    L.append(f"{r.id} & {esc(nm)} & {N} & {dd} & {K} & {f3(r['GW-HAC'])} & {f3(r['FL-U8'])} & {f3(r.KM)} & {f3(r['GW-nw'])} & {'$\\circ$' if u[i] else ''}\\\\")
L.append(r"\end{longtable}")
# ---- factor effects
fe=json.load(open(R+"tables/v2_dev_summary_dev.json"))["factor_effects"]
name={"rule:nearest-interval":"Nearest value $-$ interval","rep:mid-native":"Midpoint $-$ SQM value","rep:mean-native":"Cell mean $-$ SQM value",
      "family:dyadic-HA-grid":"Dyadic $-$ HA-grid","family:HA-theta-HA-grid":"HA-$\\theta$ $-$ HA-grid"}
F=[r"\begin{table}[h]\centering\small\caption{Factor effects in the factorial design (development datasets): paired held-out ARI differences.}\label{tab:s-factor}",
   r"\begin{tabular}{lcrrrc}\toprule Contrast & Level & $n$ & median [95\% CI] & $p$ & W/L\\\midrule"]
prev=None
for k,v in fe.items():
    key,l=k.rsplit(":",1)
    if prev and key!=prev: F.append(r"\midrule")
    F.append(f"{name[key] if key!=prev else ''} & {l[1:]} & {v['n']} & ${v['median_diff']:+.4f}$ [${v['ci_lo']:+.4f}$, ${v['ci_hi']:+.4f}$] & {v['p_two_sided']:.2f} & {v['wins']}/{v['losses']}\\\\")
    prev=key
F.append(r"\bottomrule\end{tabular}\end{table}")
# ---- configurations at level 3
b=pd.read_csv(R+"tables/v2_factorial_dev.csv")
cf=b[b.level==3].copy()
fam={"HA-grid":"HA-grid","HA-theta":"HA-$\\theta$","dyadic":"dyadic","Beta":"Beta warp","Quantile":"nested quantile"}
rp={"native":"SQM / native","mid":"midpoint","mean":"cell mean"}
C=[r"\begin{table}[h]\centering\small\caption{Word-domain configurations at level 3 (development datasets).}\label{tab:s-config}",
   r"\begin{tabular}{lllrr}\toprule Family & Representative & Assignment & median ARI & GW-HAC $-$ config.\\\midrule"]
for _,r in cf.sort_values("method").iterrows():
    a,bb,cc=r.method.split("|")
    C.append(f"{fam[a]} & {rp[bb]} & {'interval' if cc=='interval' else 'nearest value'} & {r.median_ari:.3f} & ${r.GWHAC3_minus_median_diff:+.3f}$\\\\")
C.append(r"\bottomrule\end{tabular}\end{table}")
# ---- decomposition
de=pd.read_csv if False else None
def tex_rows(path):
    t=open(path).read(); body=t.split(r"\midrule")[1].split(r"\bottomrule")[0].strip(); return body
D_=[r"\begin{table}[h]\centering\small\caption{Objective decomposition on the $k$-means partition (development datasets, medians).}\label{tab:s-decomp}",
    r"\begin{tabular}{lrrrrrr}\toprule Family & Level & $(J_\ell-J_{\mathrm N})/J_{\mathrm N}$ & $|J_\ell-J_{\mathrm N}|/J_{\mathrm N}$ & $N\hat D_\ell/J_{\mathrm N}$ & $|T_2|/(T_1+T_3)$ & excess\\\midrule",
    tex_rows(R+"tables/v2_decomposition_dev.tex").replace("HA-theta","HA-$\\theta$"),r"\bottomrule\end{tabular}\end{table}"]
TF=[r"\begin{table}[h]\centering\small\caption{Feature-level checks by level (development datasets).}\label{tab:s-feat}",
    r"\begin{tabular}{lrrrr}\toprule Quantity & $\ell=2$ & $\ell=3$ & $\ell=4$ & $\ell=5$\\\midrule"]
tf=[l.split("&") for l in tex_rows(R+"tables/v2_theory_features_dev.tex").split("\\\\") if l.strip()]
tf=np.array([[x.strip() for x in r] for r in tf])
lbl=["share of $\\hat\\alpha_j\\in[0.45,0.55]$","sign agreement of $\\hat\\alpha_j-\\tfrac12$ and $-\\hat Q_j$","features with $\\hat\\alpha_j\\ne\\tfrac12$",
     "held-out distortion reduction vs.\\ dyadic: HA-$\\theta$","\\quad HA-grid (calibrated at $\\ell=4$)","\\quad HA-grid (calibrated at $\\ell$)",
     "median price: dyadic","\\quad HA-$\\theta$","\\quad HA-grid (calibrated at $\\ell=4$)","\\quad HA-grid (calibrated at $\\ell$)","minimum price over features"]
for j,lb in enumerate(lbl): TF.append(lb+" & "+" & ".join(tf[:,j+1])+"\\\\")
TF.append(r"\bottomrule\end{tabular}\end{table}")
ce=tex_rows(R+"tables/v2_certificates_dev.tex").replace("KM &","$k$-means &")
CE=[r"\begin{table}[h]\centering\small\caption{Translation certificates (development datasets, medians).}\label{tab:s-cert}",
    r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrr}\toprule Numeric model & Family & Level & certified & certified (data-indep.) & changed & agree & agree (cert.) & agree (not cert.) & ARI vs.\ numeric & violations & violations (data-indep.)\\\midrule",
    ce,r"\bottomrule\end{tabular}}\end{table}"]
gr=tex_rows(R+"tables/v2_granularity_dev.tex")
GR=[r"\begin{table}[h]\centering\small\caption{Certified granularity of GW-HAC (development datasets).}\label{tab:s-gran}",
    r"\begin{tabular}{rrrrrr}\toprule Level & $m$ & certified & kept & violations & share with $m^\ast=m$\\\midrule",gr,r"\bottomrule\end{tabular}\end{table}"]
it=[l.split("&") for l in tex_rows(R+"tables/v2_interp_dev.tex").split("\\\\") if l.strip()]
km={"consistency_proto":"projection vs.\\ refit: prototypes","consistency_sample":"projection vs.\\ refit: samples","desc_words_per_cluster":"distinguishing words per cluster",
    "features_for_90pct":"features for 90\\% fidelity","features_total":"features in total","fidelity":"fidelity of descriptions","tree_depth":"tree depth"}
IT=[r"\begin{table}[h]\centering\small\caption{Description quality (development datasets, medians).}\label{tab:s-interp}",r"\begin{tabular}{llrr}\toprule Quantity & Method & Level & Value\\\midrule"]
for r in it:
    k,m,l,v=[x.strip() for x in r]; IT.append(f"{km[k]} & {m.replace('HA-theta','HA-$\\theta$').replace('KM+HA-words','$k$-means + HA words')} & {'--' if l=='0' else l} & {v}\\\\")
IT.append(r"\bottomrule\end{tabular}\end{table}")
no=json.load(open(R+"tables/v2_dev_summary_dev.json"))["noise_median_flip"]
NO=[r"\begin{table}[h]\centering\small\caption{Median share of changed assignments under Gaussian noise (development datasets).}\label{tab:s-noise}",
    r"\begin{tabular}{lrrr}\toprule Method & $\sigma=0.01$ & $\sigma=0.02$ & $\sigma=0.05$\\\midrule"]
for m_ in ("GW-HAC","dyadic","FL-U8","KM"):
    NO.append(f"{ {'KM':'$k$-means','dyadic':'GW-HAC, dyadic'}.get(m_,m_)} & "+" & ".join(f"{no[s][m_]:.3f}" for s in ("0.01","0.02","0.05"))+"\\\\")
NO.append(r"\bottomrule\end{tabular}\end{table}")
STG=[("calib_grid","HA-grid calibration"),("calib_theta","HA-$\\theta$ calibration"),("semantize","semantization"),("fit_gwhac","GW-HAC fit (best of 5)"),
     ("GW-HAC total","GW-HAC total"),("FL-U8","FL-U8"),("KM","$k$-means"),("GW-nw","GW-nw"),("IMM+ExKMC","IMM + ExKMC")]
TI=[r"\begin{table}[h]\centering\small\caption{Running time in milliseconds on one core: median over datasets, with the median ratio to $k$-means.}\label{tab:s-time}",
    r"\begin{tabular}{lrrrr}\toprule Stage & development & ratio & confirmatory & ratio\\\midrule"]
tp=R+"tables/timing_summary.csv"
if os.path.exists(tp):
    t=pd.read_csv(tp)
    def cell(g,st,col):
        x=t[(t.group==g)&(t.stage==st)]
        if x.empty: return "--"
        v=float(x[col].iloc[0])
        if col!="median_s": return f"{v:.2f}"
        v*=1000
        return f"{v:.0f}" if v>=100 else f"{v:.1f}" if v>=1 else f"{v:.2f}"
    for st,lb in STG:
        if (t.stage==st).any():
            TI.append(f"{lb} & {cell('dev',st,'median_s')} & {cell('dev',st,'median_ratio_to_KM')} & {cell('val2',st,'median_s')} & {cell('val2',st,'median_ratio_to_KM')}\\\\")
else:
    TI+=[f"{lb} & \\pending{{}} & \\pending{{}} & \\pending{{}} & \\pending{{}}\\\\" for _,lb in STG[:8]]
TI.append(r"\bottomrule\end{tabular}\end{table}")
doc=r"""\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}\usepackage{amsmath,amssymb,booktabs,longtable,xcolor,graphicx}
\usepackage[section]{placeins}
\newcommand{\pending}[1]{\textcolor{red}{[pending#1]}}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary Material for ``High-Resolution Theory of Hedge-Algebra Quantization, with Application to Multigranular Word Clustering''}
\author{Nguyen Duy Hieu and Hoang Trong Minh}\date{}
\begin{document}\maketitle
This supplement lists the datasets and gives the development results that the paper reports only in summary. All ARI values are on held-out data and averaged over splits. Abbreviations and methods are as defined in Section~7 of the paper. Every number is generated from the released result files.

\section{Datasets}
Table~\ref{tab:s-dev} lists the 25 development datasets and Table~\ref{tab:s-val11} the 11 preliminary datasets, with the number of samples $N$, numeric features $d$, classes $K$ and splits. Table~\ref{tab:s-confirm} lists the 126 confirmatory datasets in the order of their OpenML ids; $d$ counts the numeric features kept, and the mark $\circ$ flags the 38 uninformative datasets on which every method has ARI below $0.02$.
"""+small_table(dev,"Development datasets: mean held-out ARI over splits.","tab:s-dev")+"\n"+small_table(val,"Preliminary datasets: mean held-out ARI over splits.","tab:s-val11")+r"""
\FloatBarrier
{\small\setlength{\tabcolsep}{4pt}
"""+"\n".join(L)+r"""}

\section{Factorial design}
Table~\ref{tab:s-factor} gives each factor contrast at levels 2--5, pooled over the other factors: the median paired difference in held-out ARI with its bootstrap 95\% CI, the two-sided Wilcoxon $p$-value and wins/losses. No median exceeds $0.0021$ in absolute value. Table~\ref{tab:s-config} gives every configuration at level 3; the last column is the median paired difference of GW-HAC (HA-grid, SQM value, interval) minus the configuration.
"""+"\n".join(F)+"\n"+"\n".join(C)+r"""
\FloatBarrier
\section{Theory checks}
Table~\ref{tab:s-feat} extends Table~3 of the paper with the HA-grid calibration at the fixed level 4 and the minimum price over features. Table~\ref{tab:s-decomp} reports the terms of Proposition~6 on the $k$-means partition; ``excess'' is $(J_{\mathrm N}(\hat{\mathcal C}_\ell)-J_{\mathrm N}(\mathcal C_{\mathrm{KM}}))/J_{\mathrm N}(\mathcal C_{\mathrm{KM}})$, the relative increase of the weighted $k$-means objective when the partition $\hat{\mathcal C}_\ell$ fitted by GW-HAC replaces the $k$-means partition $\mathcal C_{\mathrm{KM}}$.
"""+"\n".join(TF)+"\n"+"\n".join(D_)+r"""
\FloatBarrier
\section{Certificates and certified granularity}
Table~\ref{tab:s-cert} reports Theorem~4 for two numeric models, EWA-HAC and $k$-means: the share of certified assignments for the data-dependent and data-independent conditions, the share changed by translation, the agreement of GW-HAC with the numeric model (overall, on certified and on uncertified samples), and the violations, which are zero throughout. Table~\ref{tab:s-gran} reports Corollary~2 for GW-HAC: the share certified $m$ levels down, the share whose projected assignment is kept, and the distribution of $m^\ast$.
"""+"\n".join(CE)+"\n"+"\n".join(GR)+r"""
\FloatBarrier
\section{Description quality and robustness}
Table~\ref{tab:s-interp} gives the fidelity of descriptions at levels 3, 2 and 1, the agreement between projecting a fine model and refitting at the coarse level, and the length of descriptions. Table~\ref{tab:s-noise} gives the share of held-out assignments that change under Gaussian noise added to the scaled data.
"""+"\n".join(IT)+"\n"+"\n".join(NO)+r"""
\FloatBarrier
\section{Running time}
Table~\ref{tab:s-time} reports wall-clock times on one core, measured by \texttt{v2/75\_timing.py}.
"""+"\n".join(TI)+r"""
\end{document}
"""
open(OUT,"w").write(doc)
