import torch

class Slicer:
  def __init__(self, atoms, modules):
    self.atoms = list(atoms)
    self.modules = modules
    print(f"Created slicer for {modules} modules (atoms {atoms})")

  def slice_atoms(self, tensor):
    return tensor[:, self.atoms]

  def slice_modules(self, modules):
    return modules[:self.modules]

def test_slicer():
  slicer = Slicer({0, 2, 3}, 2)
  preds = torch.rand((100, 5))
  
  assert (slicer.slice_atoms(preds) == preds[:, [0, 2, 3]]).all()
  assert slicer.slice_modules([0, 2, 4, 6, 8]) == [0, 2]

