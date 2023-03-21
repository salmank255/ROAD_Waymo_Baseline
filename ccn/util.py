import torch
import pytest
import matplotlib.pyplot as plt
import numpy as np

def train(dataloader, model, clayer, loss_fn, optimizer, device, ratio=1.):
    size = len(dataloader.dataset)
    model, clayer = model.to(device), clayer.to(device)
    slicer = clayer.slicer(ratio)
    model.train()

    for batch, (X, y) in enumerate(dataloader):   
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        constrained = clayer(pred, goal=y, slicer=slicer)

        constrained, y = slicer.slice_atoms(constrained), slicer.slice_atoms(y)
        loss = loss_fn(constrained, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

@pytest.mark.skip(reason="this is not a test")
def test(dataloader, model, clayer, loss_fn, device):
    size = len(dataloader.dataset)
    model, clayer = model.to(device), clayer.to(device)
    model.eval()

    test_loss = 0.
    correct = 0.
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            pred = clayer(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.where(pred > 0.5, 1., 0.) == y).sum(dim=0)

    test_loss /= size
    correct /= size

    correct = [100 * rate for rate in correct]
    accuracy = ", ".join([f"{rate:>0.1f}%" for rate in correct])
    print(f"Test Error: \n Accuracy: {accuracy}")
    print(f" Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

def draw_classes(model, draw=None, path=None, device='cpu', show=False):
    dots = np.arange(0., 1., 0.001, dtype = "float32")
    grid = torch.tensor([(x, y) for y in dots for x in dots]).to(device)
    model = model.to(device)
    preds = model(grid).detach()

    classes = preds.shape[1]
    fig, ax = plt.subplots(1, classes, figsize=(20, 20 * classes))
    for i, ax in enumerate(ax):
        image = preds[:, i].view((len(dots), len(dots))).to('cpu')
        # ax.imshow(
        #     image, 
        #     cmap='hot', 
        #     interpolation='nearest', 
        #     origin='lower', 
        #     extent=(0., 1., 0., 1.),
        #     vmin=0.,
        #     vmax=1.
        # )
        ax.contourf(
            dots,
            dots,
            image, 
            cmap='hot', 
            origin='lower', 
            extent=(0., 1., 0., 1.),
            vmin=0.1,
            vmax=1.
        )
        if draw != None: draw(ax, i)    

    if show:
        plt.show()

    if not path is None:
        plt.savefig(path)
        plt.close()

    return fig



