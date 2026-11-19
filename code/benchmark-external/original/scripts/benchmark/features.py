"""Fixed single-composition descriptors; no dataset-fitted featurization."""
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty,Stoichiometry
from .design import OUT,write

COMPONENT_LEVELS=['bulk','total','ambiguous','(not clearly mentioned, probably total)',
                  'permeation-derived partial oxide-ion conductivity']
COMPONENT_COLUMNS=['component_bulk','component_total','component_ambiguous',
                   'component_probably_total_unverified','component_permeation_other']


def featurizers():
    return ElementProperty.from_preset('magpie',impute_nan=False),Stoichiometry(p_list=[0,2,3,5,7,10],num_atoms=False)


def one_record(row,magpie,stoich):
    if row['normalization_status']!='confident_numeric_nominal':
        values=[np.nan]*(len(magpie.feature_labels())+len(stoich.feature_labels())+1)
    else:
        comp=Composition(row['Formula'])
        values=list(magpie.featurize(comp))+list(stoich.featurize(comp))+[comp.get_atomic_fraction('O')]
    assert row['final_component'] in COMPONENT_LEVELS
    values += [float(row['final_component']==c) for c in COMPONENT_LEVELS]
    return values


def generate(rows):
    magpie,stoich=featurizers()
    labels=list(magpie.feature_labels())+list(stoich.feature_labels())+['oxygen_atomic_fraction']+COMPONENT_COLUMNS
    assert len(set(labels))==len(labels)
    dictionary=[]
    for label in labels:
        family='Magpie fixed elemental-property statistics' if label in magpie.feature_labels() else \
            'matminer stoichiometric p-norm' if label in stoich.feature_labels() else \
            'oxygen atomic fraction' if label=='oxygen_atomic_fraction' else 'fixed component one-hot indicator'
        dictionary.append(dict(feature=label,definition=family,
            source='matminer 0.9.3 / pymatgen; fixed Magpie elemental lookup' if not label.startswith('component_') else 'frozen final_component annotation',
            fitted_globally=False,missing_policy='NaN retained until training-fold median imputation',
            note='No oxidation-state inference, Delta_O, prefactor, HT, family or vacancy features. Composition fractions are per-record identities, not learned normalization.'))
    frame=pd.DataFrame([one_record(r,magpie,stoich) for r in rows],
        index=pd.Index([r['row_id'] for r in rows],name='row_id'),columns=labels)
    frame=frame.replace([np.inf,-np.inf],np.nan)
    frame.to_csv(OUT/'composition_features.csv',float_format='%.17g')
    write('feature_dictionary.csv',dictionary)
    write('feature_generation_flags.csv',[dict(row_id=r['row_id'],normalization_status=r['normalization_status'],
        nan_features=int(frame.loc[r['row_id']].isna().sum()),component=r['final_component']) for r in rows])
    return frame


def load():
    return pd.read_csv(OUT/'composition_features.csv',index_col='row_id',float_precision='round_trip')
