import numpy as np

from . import ConstraintsGroup
from .literal import Literal


def get_all_eligible_literals(r_to_explore, r_to_exclude):
    all_eligible_literals = []
    atoms_to_explore = [lit.atom for lit in r_to_explore.body]
    atoms_to_exclude = [lit.atom for lit in r_to_exclude.body]
    for atom in atoms_to_explore:
        if atom not in atoms_to_exclude:
            all_eligible_literals.append(atom)
    return all_eligible_literals


def get_max_ranking(all_eligible_literals, literal_ranking):
    positions = [literal_ranking.index(lit) for lit in all_eligible_literals]
    return all_eligible_literals[np.argmax(positions)]


def create_new_rule(original_rule, extra_literal, positive):
    # TODO: conj or disj
    new_literal = Literal(atom=extra_literal, positive=positive)
    original_rule.body.add(new_literal)
    return original_rule


def extend_rules_set(R, R_other, literal_ranking):
    rules_to_add_to_R = [None]
    while len(rules_to_add_to_R) > 0:
        rules_to_add_to_R = []
        rules_to_remove_from_R = []
        for r in R:
            for r_other in R_other:
                # select literal from r_other to connect to r rule:
                # l = max_lambda over literals in either body of r or body of r_other
                all_eligible_literals = get_all_eligible_literals(r_other, r)
                if len(all_eligible_literals) == 0:
                    continue
                max_ranking_body_literal = get_max_ranking(all_eligible_literals, literal_ranking)

                # now add new rules:
                rules_to_add_to_R.append(create_new_rule(r, max_ranking_body_literal, positive=True))
                rules_to_add_to_R.append(create_new_rule(r, max_ranking_body_literal, positive=False))

                # finally remove rules:
                rules_to_remove_from_R.append(r)

        # remove found rules
        for rule in rules_to_remove_from_R:
            R.remove(rule)
        # add newly-created rules
        R.extend(rules_to_add_to_R)
    return R


def strong_coherency_constraint_preprocessing(R_atom, literal_ranking):
    R_atom_plus, R_atom_minus = [], []
    for constr in R_atom:
        if constr.head.positive:
            R_atom_plus.append(constr)
        else:
            R_atom_minus.append(constr)
    print(R_atom_minus, R_atom_plus)

    R_atom_plus = extend_rules_set(R=R_atom_plus, R_other=R_atom_minus, literal_ranking=literal_ranking)
    R_atom_minus = extend_rules_set(R=R_atom_minus, R_other=R_atom_plus, literal_ranking=literal_ranking)

    R_atom = ConstraintsGroup(R_atom_plus.extend(R_atom_minus))
    return R_atom
