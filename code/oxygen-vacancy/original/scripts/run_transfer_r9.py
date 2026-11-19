from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts/phase3b'))
from common import *
from run import execute
from downstream import baseline_job,augmented_job
from auxiliary_reuse_r9 import auxiliary_job
if __name__=='__main__':
 p=plan();configs=read('experimental_configurations.json');tasks=[(c,m) for c in configs for m in c['downstream_models']]
 for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,f
 execute(baseline_job,tasks,'baseline_selection');execute(auxiliary_job,read('auxiliary_jobs.json'),'auxiliary_fitting');execute(augmented_job,tasks,'matched_downstream_fitting')
 js(OUT/'fit_completed.json',{'outer_configurations':len(configs),'R9_current_FE':True,'all_20_shuffle_plus_20_random_seeds_retained':True,'no_scoring_during_selection':True})
