import numpy as np
import logging

def optimize_with_torch(E1,
                        E2,
                        O,
                        weight,
                        initial,
                        spectrum=None,
                        spectrum_mode=None,
                        max_iter=200,
                        tol=1e-5,
                        kappa=3.0,
                        gpu=False,
                        ):
    r"""
    Find a set of coefficients x for the electric field, such that the intensity square |Ex|^2 matches the observation.
    This function uses PyTorch for this task.

    Mathematical details follow.

    Assume the electric field $E(t) = \frac{1}{\sqrt{N}} \sum_i x(\omega_i) exp(-i \omega_i t)$.

    Given:
      a. $O(W, \theta) = |E x|^2$,

    Optimization goal:
    Minimize
       $f(x) + g(y)$,

    where:
      $f(x) = 1/2 x^2$
      $g(y) = 1/2 \sum (y - O)^2$

    s.t.
      $y = h(x) = |E1 x|^2 + |E2 x|^2$.

    Returns: the complex values of the coefficients of the FEL electric field, in time and in energy.
    """
    import torch
    from torch.optim import SGD, Adam

    # the inputs have been optimized for Fortran
    # the last axis is Up, so shift it first, so that the axes mean [Up, Obs, spectrum]
    mE1 = np.moveaxis(E1,-1,0)
    mE2 = np.moveaxis(E2,-1,0)
    # C array for speed
    mE1 = np.ascontiguousarray(mE1)
    mE2 = np.ascontiguousarray(mE2)
    n_Up = mE1.shape[0]

    # dimension of the spectrum in complex space
    Ndim = len(initial)//2

    ratio_bins = len(O)/Ndim

    # the initial solution is the same for each Up
    initial = np.stack(n_Up*[initial], axis=0)
    # model representing the simulation step
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.from_numpy(initial))
            self.E1 = torch.nn.Parameter(torch.from_numpy(mE1), requires_grad=False)
            self.E2 = torch.nn.Parameter(torch.from_numpy(mE2), requires_grad=False)
        def forward(self, data=None):
            # for each Up, multiply (E1 x)^2 + (E2 x)^2 and normalize
            y = (torch.bmm(self.E1, self.x.unsqueeze(-1))**2 + torch.bmm(self.E2, self.x.unsqueeze(-1))**2).squeeze(-1)
            return y/torch.sqrt(torch.sum(y**2, dim=-1, keepdims=True))
    # loss function, compatible with Fortran code
    # weights of the loss term
    # the Fortran code normalizes by L to ensure convergence, but this is inefficient here
    L = 1.0
    sigma = 0.95/L
    tau = 0.0
    if spectrum_mode == 1:
        sigma = 0.95/L
        tau = 0.95/L

    # flatten target
    s = torch.from_numpy(O.flatten()).unsqueeze(0)
    w = torch.from_numpy(weight.flatten()).unsqueeze(0)

    Nobs = s.shape[-1]

    # this models 1/2 sigma |observation - prediction|^2 + tau
    class StreakingLossFunction(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = w
        def forward(self, output, target):
            return 0.5*torch.mean((self.w**2) * (output - target)**2, dim=-1)

    class SpectrumLossFunction(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, output, target):
            return 0.5*torch.mean((output - target)**2, dim=-1)*2

    # do gradient descent and check for convergence
    target_spectrum = torch.from_numpy(spectrum)

    if gpu:
        target_spectrum = target_spectrum.cuda()
        s = s.cuda()
        w = w.cuda()

    model = Model()
    streaking_loss_fn = StreakingLossFunction(w)
    spectrum_loss_fn = SpectrumLossFunction()
    if gpu:
        model = model.cuda()
        streaking_loss_fn = streaking_loss_fn.cuda()
        spectrum_loss_fn = spectrum_loss_fn.cuda()

    #optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=1e-3)
    err = np.ones(100)
    converged = False
    evolution = np.zeros((max_iter, n_Up), dtype=np.float64)
    evolution_spec = np.zeros((max_iter, n_Up), dtype=np.float64)
    evolution_X = np.zeros((max_iter, n_Up, model.x.shape[-1]), dtype=np.float64)
    for i in range(max_iter):
        # zero gradient
        optimizer.zero_grad()
        # simulate angular streaking
        y = model()
        # calculate mean-squared error
        streak_loss = streaking_loss_fn(y, s)
        spectrum_loss = 0
        # if required add the loss for the spectrum
        predicted_spectrum = model.x[:, :Ndim]**2 + model.x[:, Ndim:]**2
        predicted_spectrum = predicted_spectrum/torch.sqrt(torch.sum(predicted_spectrum, dim=-1, keepdims=True))
        predicted_spectrum = torch.sqrt(1e-20 + predicted_spectrum)
        spectrum_loss = spectrum_loss_fn(predicted_spectrum, target_spectrum)
        # final loss is a linear combination of losses:
        loss = sigma*streak_loss + (tau/kappa)*ratio_bins*spectrum_loss
        # minimize loss for all Up, by averaging over Up
        all_loss = torch.mean(loss, dim=0)
        all_loss.backward()
        optimizer.step()
        # display current state
        if i % 10 == 0:
            logging.info(f"Step {i}: {all_loss.item()}")
        # get current loss and cache it
        err[i%100] = all_loss.cpu().item()
        evolution[i,:] = 2*streak_loss.cpu().detach().numpy()
        evolution_spec[i,:] = 2*spectrum_loss.cpu().detach().numpy()
        evolution_X[i,:,:] = model.x.cpu().detach().numpy()
        # check for convergence
        if i > 100:
            if np.fabs(err[99] - err[0]) < tol:
                converged = True
                evolution[i:,:] = evolution[i,None,:]
                evolution_spec[i:,:] = evolution_spec[i,None,:]
                break
    # get the best Up from the final loss
    L = loss.cpu().detach().numpy()
    idx_A = np.argmin(L)
    # return the Up index, the x result and the prediction
    x = model.x[idx_A,:].cpu().detach().numpy()
    pred = model()[idx_A,:].cpu().detach().numpy()
    evolution = evolution[:,idx_A]
    evolution_spec = evolution_spec[:,idx_A]
    evolution_X = evolution_X[:,idx_A,:]
    return idx_A, converged, x, pred, evolution, evolution_spec, evolution_X

