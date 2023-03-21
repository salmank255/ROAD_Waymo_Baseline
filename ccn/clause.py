from ast import literal_eval
import numpy as np
from .literal import Literal
from .constraint import Constraint

class Clause:
    def __init__(self, literals):
        if isinstance(literals, str):
            # Clause(string)
            literals = [Literal(lit) for lit in literals.split(' ')]
            self.literals = frozenset(literals)
        else:
            # Clause([Literals])
            self.literals = frozenset(literals)


    def __len__(self):
        return len(self.literals)

    def __iter__(self):
        return iter(self.literals)

    def __eq__(self, other):
        if not isinstance(other, Clause): return False
        return self.literals == other.literals

    def __hash__(self):
        return hash(self.literals)

    def __str__(self):
        return ' '.join([str(literal) for literal in sorted(self.literals)])

    @classmethod 
    def from_constraint(cls, constraint):
        body = [lit.neg() for lit in constraint.body]
        return cls([constraint.head] + body)

    @classmethod 
    def random(cls, num_classes):
        atoms_count = np.random.randint(low=1, high=num_classes, size=1)
        atoms = np.random.randint(num_classes, size=atoms_count)
        
        pos = atoms[np.random.randint(2, size=atoms_count) == 1]
        literals = [Literal(atom, atom in pos) for atom in atoms]
        return cls(literals)

    def shift_add_n0(self):
        n0 = Literal(0, False)
        return Clause([Literal(lit.atom + 1, lit.positive) for lit in self] + [n0])

    def fix_head(self, head):
        if not head in self.literals:
            raise Exception('Head not in clause')
        body = [lit.neg() for lit in self.literals if lit != head]
        return Constraint(head, body)

    def always_true(self):
        for literal in self.literals:
            if literal.neg() in self.literals:
                return True 
        return False

    def resolution_on(self, other, literal):
        result = self.literals.union(other.literals).difference({literal, literal.neg()})
        result = Clause(result)
        return None if result.always_true() else result

    def resolution(self, other, literal=None):
        if literal != None:
            return self.resolution_on(other, literal)

        for lit in self.literals:
            if lit.neg() in other.literals:
                return self.resolution_on(other, lit)

        return None

    def always_false(self):
        return len(self) == 0

    def coherent_with(self, preds):
        pos = [lit.atom for lit in self.literals if lit.positive]
        neg = [lit.atom for lit in self.literals if not lit.positive]

        preds = np.concatenate((preds[:, pos], 1 - preds[:, neg]), axis=1)
        preds = preds.max(axis=1)
        return preds > 0.5        

    def is_subset(self, other):
        return self.literals.issubset(other.literals)

    def atoms(self):
        return {lit.atom for lit in self.literals}

def test_eq():
    assert Clause('1 n2 1 2') == Clause('2 1 n2')
    assert Clause('1 n2 3 n4') != Clause('1 n2 3 4') 

def test_str():
    assert str(Clause('1 n2 1 2')) == '1 n2 2'
    assert str(Clause([Literal('1'), Literal('n2'), Literal('1'), Literal('2')])) == '1 n2 2'

def test_shift_add_n0():
    assert Clause('0 n1 2 n3').shift_add_n0() == Clause('n0 1 n2 3 n4')

def test_always_true():
    assert not Clause('1 2 n3').always_true()
    assert Clause('1 2 n3 n1').always_true()

def test_constraint():
    assert Clause('1 2 n3').fix_head(Literal('1')) == Constraint('1 :- n2 3')
    assert Clause('1 2 n3').fix_head(Literal('1')) != Constraint('n1 :- n2 3') 
    assert Clause.from_constraint(Constraint('2 :- 1 n0')) == Clause('2 n1 0')
    assert Clause.from_constraint(Constraint('n2 :- 1 n0')) != Clause('2 n1 0')

def test_resolution():
    c1 = Clause('1 n2 3')
    c2 = Clause('2 4 n5')
    assert c1.resolution(c2) == Clause('1 3 4 n5')
    c1 = Clause('1 2 n3')
    c2 = Clause('n1 2 3')
    assert c1.resolution(c2) == None 
    c1 = Clause('1 2 n3')
    c2 = Clause('n3 n4 5 6')
    assert c1.resolution(c2) == None

def test_coherent_with():
    c = Clause('0 1 n2 n3')
    preds = np.array([ 
        [.1, .2, .8, .9, .1],
        [.2, .6, .6, .7, .2],
        [.2, .3, .2, .8, .3],
        [.6, .3, .3, .6, .4],
        [.9, .9, .1, .1, .5],
        [.4, .5, .6, .7, .6]
    ])

    assert (c.coherent_with(preds) == [False, True, True, True, True, False]).all()

def test_random():
    c = [Clause.random(10) for i in range(10)]
    assert not (np.array(c) == c[0]).all() 

def test_is_subset():
    c1 = Clause('1 2 n3 4')
    c2 = Clause('1 n3')
    c3 = Clause('1 3')
    assert not c1.is_subset(c2)
    assert c2.is_subset(c1)
    assert not c3.is_subset(c1)

def test_atoms():
    assert Clause('1 3 n5 17').atoms() == {1, 3, 5, 17}
