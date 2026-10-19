"""Cross-check every quantitative claim in main.tex against the frozen analysis files."""
import json, collections, re, sys
import os
R=(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))+'/'  # repo root; default assumes white_papers/GLEE_revision/
d=json.load(open(R+'data/rsi_bench/glee_pia_analysis.json'))
ra=json.load(open(R+'data/rsi_bench/author_ratings_analysis.json'))
rep=json.load(open(R+'data/rsi_bench/glee_replay_summary.json'))
rows=[json.loads(l) for l in open(R+'data/rsi_bench/glee_replay_scored.jsonl')]
sc=[json.loads(l) for l in open(R+'data/rsi_bench/glee_pia_stratified_scored.jsonl')]
out=[]; bad=[]
def chk(name,paper,got,tol=0.0006):
    ok = abs(paper-got)<=tol if isinstance(paper,(int,float)) and isinstance(got,(int,float)) else paper==got
    out.append((name,paper,got,'OK' if ok else 'MISMATCH'))
    if not ok: bad.append(name)
A=d['aggregates']; W=A['with_degenerate']; X=A['without_degenerate']; ci=d['game_level_bootstrap']['ci_95']['kappa_overall']
for fam,(a,x,lo,hi) in {'bargaining':(-0.013,-0.023,-0.118,0.105),'negotiation':(-0.029,-0.067,-0.149,0.054),'persuasion':(0.170,0.112,0.070,0.154),'overall':(0.043,0.007,-0.043,0.079)}.items():
    chk(f'T1 {fam} all',a,W['kappa_overall'][fam]['unweighted_mean']); chk(f'T1 {fam} excl',x,X['kappa_overall'][fam]['unweighted_mean'])
    chk(f'T1 {fam} CI lo',lo,ci[fam]['low']); chk(f'T1 {fam} CI hi',hi,ci[fam]['high'])
S=d['shuffled_cluster_floor']
for fam,(m,l,h) in {'bargaining':(0.032,-0.003,0.072),'negotiation':(0.018,-0.020,0.060),'persuasion':(0.179,0.156,0.204)}.items():
    s=S['kappa_overall'][fam]; chk(f'T1 shuffled {fam} mean',m,s['shuffled_mean']); chk(f'T1 shuffled {fam} lo',l,s['ci_low']); chk(f'T1 shuffled {fam} hi',h,s['ci_high'])
# counts below shuffled mean
cells=[(dim,f,v) for dim,fv in S.items() for f,v in fv.items() if v.get('real') is not None and v.get('n_valid_perms',0)>0]
chk('3-judge cells',14,len(cells)); chk('3-judge below mean',13,sum(v['real']<v['shuffled_mean'] for _,_,v in cells)); chk('3-judge above 95%',0,sum(v['real']>v['ci_high'] for _,_,v in cells))
S2=d['leave_one_judge_out']['shuffled_cluster_floor']
c2=[(dim,f,v) for dim,fv in S2.items() for f,v in fv.items() if v.get('real') is not None and v.get('n_valid_perms',0)>0]
chk('2-judge cells',14,len(c2)); chk('2-judge below mean',14,sum(v['real']<v['shuffled_mean'] for _,_,v in c2))
L=d['leave_one_judge_out']['aggregates']
chk('LOO all 0.30',0.30,L['with_degenerate']['unweighted_mean'],0.005) if 'unweighted_mean' in L['with_degenerate'] else None
# LOO structure: with_degenerate -> dim -> scope? fall back
def loo(which):
    v=L[which]; 
    return v['kappa_overall']['overall']['unweighted_mean'] if 'kappa_overall' in v else v['unweighted_mean']
chk('LOO all',0.30,loo('with_degenerate'),0.005); chk('LOO excl',0.25,loo('without_degenerate'),0.005)
O=d['offset_robust_agreement']; icc=[v['icc_3_1_consistency'] for dm in O.values() for v in dm.values() if v.get('icc_3_1_consistency') is not None]
al=[v['krippendorff_alpha_ordinal'] for dm in O.values() for v in dm.values() if v.get('krippendorff_alpha_ordinal') is not None]
chk('ICC min',0.28,min(icc),0.005); chk('ICC max',0.70,max(icc),0.005); chk('alpha min excl pers-outcome',0.13,min(v for v in al if v>0),0.005); chk('alpha max',0.63,max(al),0.005); chk('alpha pers outcome',-0.04,O['outcome_consistency']['persuasion']['krippendorff_alpha_ordinal'],0.005)
V=d['variance_decomposition']; cn=[(dm,f,v) for dm,fv in V.items() for f,v in fv.items() if f!='overall' and v.get('real_eta_squared') is not None and v.get('n_valid_perms',0)>0]
chk('eta cells',11,len(cn)); chk("eta above shuffled hi",6,sum(v['real_eta_squared']>v['ci_high'] for _,_,v in cn))
chk('eta neg horizon',0.52,V['horizon_strategy_planning']['negotiation']['real_eta_squared'],0.005); chk('eta neg horizon shuffled',0.15,V['horizon_strategy_planning']['negotiation']['shuffled_mean'],0.005)
JA=d['judge_agreement']
# leniency of LG vs other two
dif=[]
for dim in ['horizon_strategy_planning','concession_handling']:
    for fam,v in JA[dim].items():
        m=v['per_judge_mean']
        if m['LiteralGroundedness'] is None or isinstance(m['LiteralGroundedness'],str): continue
        o=[m['GameTheoreticRigor'],m['OpponentResponsiveness']]; dif += [m['LiteralGroundedness']-x for x in o]
chk('LG lenient min (horizon/concession)',0.6,min(dif),0.06); chk('LG lenient max (horizon/concession)',1.4,max(dif),0.06)
m=JA['valuation_reasoning']['bargaining']['per_judge_mean']; chk('LG valuation bargaining max diff',0.8,max(m['LiteralGroundedness']-m['GameTheoreticRigor'],m['LiteralGroundedness']-m['OpponentResponsiveness']),0.06)
p=JA['horizon_strategy_planning']['negotiation']['pairwise']['GameTheoreticRigor|LiteralGroundedness']; chk('spearman neg horizon',0.71,p['spearman_rho'],0.005); chk('cohen neg horizon',-0.14,p['cohens_kappa'],0.005)
chk('bargaining outcome 5s',0.59,JA['outcome_consistency']['bargaining']['frac_eq_5'],0.006); chk('persuasion outcome 5s',0.90,JA['outcome_consistency']['persuasion']['frac_eq_5'],0.006)
H=d['membership_homogeneity']; chk('neg round modal',0.99,H['negotiation']['mean_round_modal_share'],0.005); chk('pers round modal',0.43,H['persuasion']['mean_round_modal_share'],0.005)
AS=d['action_shuffled_cluster_floor']
for fam,(r_,m_,l_,h_) in {'bargaining':(0.103,0.122,0.092,0.157),'negotiation':(0.060,0.066,0.048,0.088),'persuasion':(0.349,0.467,None,None)}.items():
    chk(f'action {fam} real',r_,AS[fam]['real']); chk(f'action {fam} shuf',m_,AS[fam]['shuffled_mean'])
    if l_ is not None: chk(f'action {fam} lo',l_,AS[fam]['ci_low']); chk(f'action {fam} hi',h_,AS[fam]['ci_high'])
C=d['action_judge_spread_correlation']; chk('rho pers',0.51,C['persuasion']['spearman_rho'],0.006); chk('p pers',0.002,C['persuasion']['spearman_p'],0.0006); chk('rho barg',0.04,C['bargaining']['spearman_rho'],0.006); chk('rho neg',-0.01,C['negotiation']['spearman_rho'],0.006)
# dependence
D=d['dependence']; 
try:
    chk('share barg',0.94,D['bargaining']['share_clusters_sharing_a_game'],0.006); chk('share neg',1.0,D['negotiation']['share_clusters_sharing_a_game'],0.006)
except Exception as e: out.append(('dependence keys',str(list(D.keys())),str(e),'CHECK'))
CC=d['cluster_components']
try: chk('components',(7,5,29),tuple(CC[f]['n_components'] for f in ['bargaining','negotiation','persuasion']))
except Exception as e: out.append(('components keys',str(list(CC.keys())),str(e),'CHECK'))
# replay
AG=rep['aggregate_by_family_temp']
chk('rep barg default',0.085,AG['bargaining']['api_default']['mean_action_spread'],0.0006); chk('rep barg t0',0.024,AG['bargaining']['t0']['mean_action_spread'],0.0006)
chk('rep neg default',0.090,AG['negotiation']['api_default']['mean_action_spread'],0.0006); chk('rep neg t0',0.028,AG['negotiation']['t0']['mean_action_spread'],0.0006)
chk('rep cost',2.32,rep['run_stats']['estimated_cost_usd'],0.006); chk('rep calls',360,rep['run_stats']['n_calls']); chk('rep fails',13,rep['run_stats']['outcome_counts']['parse_failure'])
# exclude state 02
import numpy as np
bs=[v for k,v in rep['by_state_temp'].items() if v['game_family']=='bargaining' and k.endswith('api_default') and not k.startswith('02_')]
b0=[v for k,v in rep['by_state_temp'].items() if v['game_family']=='bargaining' and k.endswith('|t0') and not k.startswith('02_')]
chk('rep barg excl default',0.030,float(np.mean([v['action_spread'] for v in bs])),0.0006); chk('rep barg excl t0',0.030,float(np.mean([v['action_spread'] for v in b0])),0.0006)
chk('obs/rep ratio',3.5,round(0.10299/float(np.mean([v['action_spread'] for v in bs])),1),0.06)
# exact divergence
by=collections.defaultdict(list)
for r in rows:
    if r['ok']: by[(r['state_id'],r['temp_label'])].append(json.dumps(r['action'],sort_keys=True))
for t in ['api_default','t0']: chk(f'exact diverge {t}',15,sum(len(set(v))>1 for (s,tt),v in by.items() if tt==t))
fails=collections.Counter(r['state_id'][:2] for r in rows if not r['ok']); chk('fails in state02',12,fails['02']); 
st02=[r for r in rows if r['state_id'].startswith('02_') and not r['ok']]; chk('state02 capped', True, all(r['usage']['output_tokens']==600 for r in st02))
# rubinstein
for temp,exp in [('api_default',(8,0.526,0.65)),('t0',(8,0.526,0.60))]:
    xs=[r for r in rows if r['state_id'].startswith('07_bargaining') and r['temp_label']==temp and r['ok']]
    cite=[r for r in xs if re.search(r'52\.6|1/\(1\+0?\.9\)|1/1\.9',r['reasoning'])]
    sh=[r['action']['alice_gain']/(r['action']['alice_gain']+r['action']['bob_gain']) for r in cite]
    chk(f'state07 {temp} cite',exp[0],len(cite)); chk(f'state07 {temp} min',exp[1],min(sh),0.001); chk(f'state07 {temp} max',exp[2],max(sh),0.001)
# fallback
tot=sum(x['n_members'] for x in sc); fb=sum(x['n_fallback'] for x in sc)
chk('members',5870,tot); chk('fallback',883,fb); chk('fallback rate',0.150,fb/tot,0.0006); chk('judged',805,sum(x['n_judged_members'] for x in sc))
for fam,e in {'bargaining':0.108,'negotiation':0.045,'persuasion':0.215}.items():
    t_=sum(x['n_members'] for x in sc if x['game_family']==fam); f_=sum(x['n_fallback'] for x in sc if x['game_family']==fam); chk(f'fb {fam}',e,f_/t_,0.0006)
chk('judge calls',2415,sum(x['usage']['n_score_member_calls'] for x in sc)); inp=sum(x['usage']['input_tokens'] for x in sc); o=sum(x['usage']['output_tokens'] for x in sc); chk('cost',17.5,inp*3/1e6+o*15/1e6,0.06)
# ratings
AT=ra['author_vs_targets']
chk('rater val rho',0.68,AT['valuation_reasoning']['overall']['judge_mean']['spearman']['rho'],0.006); chk('rater val lo',0.43,AT['valuation_reasoning']['overall']['judge_mean']['spearman']['ci_low'],0.006); chk('rater val hi',0.84,AT['valuation_reasoning']['overall']['judge_mean']['spearman']['ci_high'],0.006)
chk('rater hor rho',0.29,AT['horizon_strategy_planning']['overall']['judge_mean']['spearman']['rho'],0.006); chk('rater hor lo',-0.20,AT['horizon_strategy_planning']['overall']['judge_mean']['spearman']['ci_low'],0.006); chk('rater hor hi',0.68,AT['horizon_strategy_planning']['overall']['judge_mean']['spearman']['ci_high'],0.006)
chk('off val',-0.87,AT['valuation_reasoning']['overall']['judge_mean']['offset']['mean_offset'],0.006); chk('off hor',-0.85,AT['horizon_strategy_planning']['overall']['judge_mean']['offset']['mean_offset'],0.006)
w=max(len(x[0]) for x in out)
for n,p,g,s in out: print(f'{n:<{w}} paper={p!s:<8} computed={g if not isinstance(g,float) else round(g,4)!s:<10} {s}')
print('\nTOTAL',len(out),'MISMATCH',len(bad),bad)
