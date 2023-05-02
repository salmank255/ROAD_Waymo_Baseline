
class Literal:
    def __init__(self, *args):
        if len(args) == 2:
            # Literal(int, bool)
            self.atom = args[0]
            self.positive = args[1]
        else:
            # Literal(string)
            plain = args[0]
            if 'n' in plain:
                self.atom = int(plain[1:])
                self.positive = False
            else:
                self.atom = int(plain)
                self.positive = True
                
    def __str__(self):
        return str(self.atom) if self.positive else 'n' + str(self.atom)

    def __hash__(self):
        return hash((self.atom, self.positive))

    def __eq__(self, other):
        if not isinstance(other, Literal): return False
        return self.atom == other.atom and self.positive == other.positive 

    def __lt__(self, other):
        if self.atom == other.atom:
            return self.positive < other.positive
        else:
            return self.atom < other.atom
            
    def neg(self):
        return Literal(self.atom, not self.positive)
    

def test_init_and_str():
    assert str(Literal('13')) == "13"
    assert str(Literal('n0')) == "n0"
    assert str(Literal('0')) == "0"

def test_neg():
    assert str(Literal('13').neg()) == 'n13'
    assert str(Literal('n0').neg()) == '0'
    assert str(Literal('0').neg()) == 'n0'

def test_comparison():
    assert Literal('13').neg() == Literal('n13')
    assert Literal('13').neg() != Literal('n14')
    assert Literal('n13') < Literal('13')
    assert Literal('13') < Literal('15')
    assert hash(Literal('14')) == hash(Literal('n14').neg())
    assert hash(Literal('14')) != hash(Literal('15'))
    assert hash(Literal('14')) != hash(Literal('n14'))

    