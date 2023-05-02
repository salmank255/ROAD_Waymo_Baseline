import numpy as np
from .literal import Literal

class Constraint:
    def __init__(self, *args):
        if len(args) == 2:
            # Constraint(Literal, [Literal])
            self.head = args[0]
            self.body = frozenset(args[1])
        else:
            # Constraint(string)
            line = args[0].split(' ')
            if line[2] == ':-':
                line = line[1:]
            assert line[1] == ':-'
            self.head = Literal(line[0])
            self.body = frozenset(Literal(lit) for lit in line[2:])

    def __eq__(self, other):
        if not isinstance(other, Constraint): return False
        return self.head == other.head and self.body == other.body

    def __lt__(self, other):
        if self.head == other.head:
            return self.body < other.body 
        else:
            return self.head < other.head

    def __hash__(self):
        return hash((self.head, self.body))
        
    def __str__(self):
        return str(self.head) + " :- " + ' '.join([str(lit) for lit in sorted(self.body)])
            
    def head_encoded(self, num_classes):
        pos_head = np.zeros(num_classes)
        neg_head = np.zeros(num_classes)
        if self.head.positive:
            pos_head[self.head.atom] = 1
        else:
            neg_head[self.head.atom] = 1
        return pos_head, neg_head
    
    def body_encoded(self, num_classes):
        pos_body = np.zeros(num_classes, dtype=int)
        neg_body = np.zeros(num_classes, dtype=int)
        for lit in self.body:
            if lit.positive:
                pos_body[lit.atom] = 1
            else:
                neg_body[lit.atom] = 1
        return pos_body, neg_body
    
    def where(self, cond, opt1, opt2):
        return opt2 + cond * (opt1 - opt2)
    
    def coherent_with(self, preds):
        num_classes = preds.shape[1]
        pos_body, neg_body = self.body_encoded(num_classes)
        pos_body = preds[:, pos_body.astype(bool)]
        neg_body = 1 - preds[:, neg_body.astype(bool)]
        body = np.min(np.concatenate((pos_body, neg_body), axis=1), axis=1)
        
        head = preds[:, self.head.atom]
        if not self.head.positive:
            head = 1 - head
            
        return body <= head

    def atoms(self):
        return {lit.atom for lit in self.body.union({self.head})}
    
def test_str():
    assert str(Constraint(Literal('1'), [Literal('n0'), Literal("2")])) == "1 :- n0 2" 
    assert str(Constraint('n0 :- 1 n2 n3')) == "n0 :- 1 n2 n3"
    assert str(Constraint('0.0 n0 :- 1 n2 n3')) == "n0 :- 1 n2 n3"

def test_eq():
    assert Constraint('0 :- n1 2 n1') == Constraint('0 :- 2 n1')
    assert Constraint('0 :- n1 2 n2') != Constraint('0 :- 2 n1')
    
def test_coherent_with():
  cons = Constraint('1 :- 0')
  assert (cons.coherent_with(np.array([
      [0.1, 0.2],
      [0.2, 0.1],
      [0.1, 0.1]
  ])) == [True, False, True]).all()

  cons = Constraint('n0 :- n1 2 3')
  assert (cons.coherent_with(np.array([
      [0.7, 0.8, 0.3, 0.4],
      [0.8, 0.8, 0.3, 0.4],
      [0.9, 0.8, 0.3, 0.4],
  ])) == [True, True, False]).all()

def test_atoms():
    c = Constraint('n2 :- 2 3 n4 5')
    assert c.atoms() == {2, 3, 4, 5}
