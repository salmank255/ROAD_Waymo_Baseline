import numpy as np
import pytest
import json
import math
import torch
from torch import nn

from .constraints_module import ConstraintsModule
from .constraints_group import ConstraintsGroup
from .constraint import Constraint
from .clauses_group import ClausesGroup 
from .slicer import Slicer
from .profiler import Profiler

class ConstraintsLayer(nn.Module):
    def __init__(self, strata, num_classes):
        super(ConstraintsLayer, self).__init__()

        # ConstraintsLayer([ConstraintsGroup], int)
        self.num_classes = num_classes
        self.atoms = nn.Parameter(torch.tensor(list(range(num_classes))), requires_grad=False)

        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)

        # Compute all strata & core 
        core = set(range(num_classes))
        strata = [stratum.heads() for stratum in strata]

        for stratum in strata:
            core = core.difference(stratum)

        assert len(core) > 0
        self.core = core
        self.strata = strata

    @classmethod
    def from_clauses_group(cls, group, num_classes, centrality):
        return cls(group.stratify(centrality), num_classes)

    def gradual_prefix(self, ratio):
        atoms = self.core
        remaining = math.floor(ratio * self.num_classes) - len(atoms)
        if (remaining <= 0): return atoms, 0

        for i, stratum in enumerate(self.strata):
            remaining -= len(stratum)
            if (remaining < 0): return atoms, i 
            atoms = atoms.union(stratum)

        return atoms, len(self.strata)

    def slicer(self, ratio):
        atoms, modules = self.gradual_prefix(ratio)
        return Slicer(atoms, modules)
    
    def to_minimal(self, tensor):
        return tensor[:, self.atoms].reshape(tensor.shape[0], len(self.atoms))

    def from_minimal(self, tensor, init):
        return init.index_copy(1, self.atoms, tensor)
        
    def forward(self, preds, goal=None, iterative=True, slicer=None):
        updated = self.to_minimal(preds)
        goal = None if goal is None else self.to_minimal(goal)

        modules = self.module_list if slicer is None else slicer.slice_modules(self.module_list)
        for module in modules:
            updated = module(updated, goal=goal, iterative=iterative)

        return self.from_minimal(updated, preds)

def test_gradual_atoms():
    groups = [ 
        ConstraintsGroup([ Constraint('n1 :- 0'), Constraint('1 :- n2') ]),
        ConstraintsGroup([ Constraint('4 :- n1'), Constraint('n3 :- 1') ])
    ]
    layer = ConstraintsLayer(groups, num_classes=5)
    assert layer.gradual_prefix(0) == ({0, 2}, 0)
    assert layer.gradual_prefix(0.33) == ({0, 2}, 0)
    assert layer.gradual_prefix(0.59) == ({0, 2}, 0)
    assert layer.gradual_prefix(0.61) == ({0, 1, 2}, 1)
    assert layer.gradual_prefix(0.66) == ({0, 1, 2}, 1)
    assert layer.gradual_prefix(1) == ({0, 1, 2, 3, 4}, 2)
    
    tensor = torch.rand(5, 5)
    slicer = layer.slicer(0.66)
    assert (slicer.slice_atoms(tensor) == tensor[:, [0, 1, 2]]).all()
    assert (slicer.slice_modules([0, 1, 2]) == [0])

def run_layer(layer, preds, backward=False):
    if backward:
        extra = torch.rand_like(preds, requires_grad=True)
        preds = preds + extra

    iter = layer(preds, iterative=True)
    tens = layer(preds, iterative=False)
    assert torch.isclose(iter, tens).all()

    if backward:
        sum = iter.sum() + tens.sum()
        sum.backward()

    return iter.detach()

def test_two_modules():
    group0 = ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    group1 = ConstraintsGroup([
        Constraint('2 :- 0'),
        Constraint('n2 :- 1'),
    ])
    group = group0 + group1 

    layer = ConstraintsLayer([group0, group1], 3)
    preds = torch.rand((5000, 3))
    updated = run_layer(layer, preds)
    assert group.coherent_with(updated.numpy()).all()

def _test_many_clauses(centrality, device, batch=1500, max_clauses=150, goals=5, backward=False):
    num_classes = 30
    goal = np.random.randint(low=0, high=2, size=(goals, num_classes))
    clauses = ClausesGroup.random(min_clauses=min(10, max_clauses), max_clauses=max_clauses, num_classes=num_classes, coherent_with=goal)
    layer = ConstraintsLayer.from_clauses_group(clauses, num_classes=num_classes, centrality=centrality)
    preds = torch.rand((batch, num_classes))

    layer, preds = layer.to(device), preds.to(device)

    updated = run_layer(layer, preds, backward=backward)
    assert clauses.coherent_with(updated.cpu().numpy()).all()
    
    if len(updated) > 100 and len(clauses) > 3:
        difs = (updated - preds).cpu()
        assert (difs == 0.).any() 
        assert (difs != 0.).any()

def _test_many_clauses_all_measures(device):
    for centrality in ClausesGroup.centrality_measures():
        _test_many_clauses(centrality, device)

def test_many_clauses_cpu():
    _test_many_clauses_all_measures('cpu')

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_many_clauses_cuda():
    _test_many_clauses_all_measures('cuda')

def test_backward_cpu():
    _test_many_clauses('katz', 'cpu', batch=10, backward=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backward_cuda():
    _test_many_clauses('katz', 'cuda', batch=10, backward=True)

def test_empty_cpu():
    _test_many_clauses('katz', 'cpu', batch=0, backward=False)
    _test_many_clauses('katz', 'cpu', batch=0, backward=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_cuda():
    _test_many_clauses('katz', 'cuda', batch=0, backward=False)
    _test_many_clauses('katz', 'cuda', batch=0, backward=True)

def test_no_clauses_cpu():
    _test_many_clauses('katz', 'cpu', batch=0, backward=False)
    _test_many_clauses('katz', 'cpu', batch=0, backward=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_clauses_cuda():
    _test_many_clauses('katz', 'cuda', max_clauses=0, backward=False)
    _test_many_clauses('katz', 'cuda', max_clauses=0, backward=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_memory():
    torch.cuda.empty_cache()
    profiler = Profiler.shared().branch("layer")
    ConstraintsModule.profiler.reset()

    device = 'cuda'
    num_classes = 42
    total_classes = num_classes + 100
    batch = 1000 

    constraints = ConstraintsGroup('../constraints/full')
    clauses = ClausesGroup.from_constraints_group(constraints)
    clauses = clauses.add_detection_label(False)
    layer = ConstraintsLayer.from_clauses_group(clauses, num_classes, 'rev-katz').to(device)

    with profiler.watch('complete'):
        with profiler.watch('preds'):
            preds = torch.rand(batch, total_classes, device=device)
        with profiler.watch('extra'):
            extra = torch.rand_like(preds, requires_grad=True)
        with profiler.watch('goal'):
            goal = torch.randint(2, preds.shape, device=device).float()
        with profiler.watch('sum'):
            summed = preds + extra
        with profiler.watch('layer'):
            result = layer(summed, goal=goal, iterative=True)
        with profiler.watch('loss'):
            result = result.sum()
        with profiler.watch('backward'):
            result.backward()

    #print(torch.cuda.memory_summary(abbreviated=True))
    
    print(json.dumps(Profiler.shared().combined(), indent=4, sort_keys=True))

    results = profiler.total(kind='gpu')
    assert results[0] <= 10



    

    