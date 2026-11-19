"""Stateless curation and leakage-safe partitions. No model or fitted preprocessing."""
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
import re
import unicodedata
from urllib.parse import unquote

ELEMENTS = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())
PAPER = 'https://www.nature.com/articles/s41597-026-07100-x#Sec4'
FAMILY_MAP = {
    'Perovskites': 'Perovskite',
    'Fluorites': 'Fluorite', 'd-Bi2O3': 'δ-Bi2O3', 'BIMEVOX': 'BIMEVOX',
    'Bi2UO6': 'Bi2UO6', 'Scheelites': 'Scheelite', 'Apatites': 'Apatite',
    'Cuspidines': 'Cuspidine', 'Melilites': 'Melilite', 'Garnets': 'Garnet',
    'Pyrochlores': 'Pyrochlore', 'Columbites': 'Columbite', 'a-SrSiO3': 'α-SrSiO3',
    **{f'Perovskite-related, {s}': 'Perovskite-related' for s in [
        'brownmillerites', 'hexagonal', 'Ruddlesden-Popper', 'Dion-Jacobson',
        'others (V-layered)', 'others (Nd-layered)', 'others (Ti-layered)',
        'others (A-site vacancies)']},
}
COMPONENT_MAP = {'bulk': 'bulk', 'total': 'total',
    '(not clearly mentioned, probably total)': 'uncertain/probably-total'}


def normalize_doi(value):
    s = unquote(value.strip()).lower()
    s = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', s)
    # Do not strip trailing punctuation: it can be a legitimate DOI character.
    return s


def normalize_formula(raw):
    """Exact rational stoichiometric ratios, no rounding or charge-balance inference.

    Returns an empty key for anything symbolic, malformed, or ambiguous.
    Numeric nominal formulas do not assert measured stoichiometry or phase identity.
    """
    s = unicodedata.normalize('NFKC', raw).strip()
    result = dict(raw_formula=raw, normalized_formula='', normalization_status='',
                  symbolic_variables_present='', notes='', element_set='')
    tokens = re.findall(r'[A-Z][a-z]?|(?:\d+(?:\.\d+)?|\.\d+)|[()]', s)
    remainder = re.sub(r'[A-Z][a-z]?|(?:\d+(?:\.\d+)?|\.\d+)|[()]', '', s)
    if remainder or not s:
        variables = sorted(set(re.findall(r'[a-zα-ωΑ-Ω]+', remainder)))
        result.update(normalization_status='symbolic_unresolved' if variables else 'unresolved',
                      symbolic_variables_present=';'.join(variables),
                      notes='No numeric substitution; composition key withheld.')
        return result
    pos = 0
    zero = False

    def amount():
        nonlocal pos, zero
        if pos < len(tokens) and re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', tokens[pos]):
            n = Fraction(tokens[pos]); pos += 1
            zero |= n == 0
            return n
        return Fraction(1)

    def group(nested=False):
        nonlocal pos
        out = defaultdict(Fraction)
        while pos < len(tokens):
            token = tokens[pos]; pos += 1
            if token == ')':
                if not nested or not out:
                    raise ValueError('Unexpected or empty parentheses')
                return out
            if token == '(':
                sub = group(True); n = amount()
                for e, v in sub.items(): out[e] += v * n
            elif token in ELEMENTS:
                out[token] += amount()
            else:
                raise ValueError('Unknown element or unexpected coefficient')
        if nested:
            raise ValueError('Unclosed parentheses')
        return out

    try:
        counts = {e: v for e, v in group().items() if v > 0}
        if not counts: raise ValueError('Empty composition')
        denominator = lcm(*(v.denominator for v in counts.values()))
        ints = {e: int(v * denominator) for e, v in counts.items()}
        divisor = reduce(gcd, ints.values())
        ints = {e: v // divisor for e, v in ints.items()}
        key = ''.join(e + (str(ints[e]) if ints[e] != 1 else '') for e in sorted(ints))
        result.update(normalized_formula=key, normalization_status='confident_numeric_nominal',
                      element_set=';'.join(sorted(ints)),
                      notes='Exact decimal ratios reduced to integers in element alphabetical order; nominal formula only, phase and true oxygen stoichiometry unverified.'
                      + (' Explicit zero coefficients omitted.' if zero else ''))
    except ValueError as exc:
        result.update(normalization_status='unresolved', notes=str(exc))
    return result


def connected_components(records):
    """Record nodes; DOI OR confident composition OR established specimen links."""
    parent = list(range(len(records)))
    edges = []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for key in ['normalized_doi', 'normalized_formula', 'established_specimen_id']:
        seen = {}
        for i, row in enumerate(records):
            v = row.get(key, '')
            if key == 'normalized_formula' and row.get('normalization_status') != 'confident_numeric_nominal':
                continue
            if not v: continue
            if v in seen:
                j = seen[v]; a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
                edges.append({'row_id_1': records[j]['row_id'], 'row_id_2': row['row_id'],
                              'link_reason': key, 'link_value': v})
            else:
                seen[v] = i
    members = defaultdict(list)
    for i in range(len(records)): members[root(i)].append(i)
    for indices in members.values():
        group_id = 'G' + records[min(indices)]['row_id'][1:]
        for i in indices:
            records[i]['connected_group_id'] = group_id
            records[i]['group_size'] = len(indices)
    return records, edges


def lofo_partition(records, family, eligible_ids=None):
    """Global provenance components determine purging even for restricted cohorts.

    All raw rows of a held-out family determine touching groups. Eligibility only
    limits endpoint test/train rows, never the scope of provenance purging.
    """
    touching = {r['connected_group_id'] for r in records if r['family'] == family}
    eligible = set(eligible_ids) if eligible_ids is not None else {r['row_id'] for r in records}
    out = {'test': [], 'purged': [], 'train': []}
    for r in records:
        if r['row_id'] not in eligible: continue
        role = 'test' if r['family'] == family else 'purged' if r['connected_group_id'] in touching else 'train'
        out[role].append(r)
    return out


def training_input_manifest(partition):
    """The ONLY prospective input to future fitted preprocessing; no target values.

    This does not fit/compute scaling, imputation, vocabulary, or feature selection.
    """
    forbidden_ids = {r['row_id'] for role in ['test','purged'] for r in partition[role]}
    forbidden_groups = {r['connected_group_id'] for role in ['test','purged'] for r in partition[role]}
    if any(r['row_id'] in forbidden_ids or r['connected_group_id'] in forbidden_groups for r in partition['train']):
        raise ValueError('Test/purged row or group supplied as preprocessing fit input')
    return [{'row_id': r['row_id'], 'connected_group_id': r['connected_group_id'],
             'normalized_formula': r['normalized_formula'],
             'conductivity_component': r['conductivity_component']}
            for r in partition['train']]
