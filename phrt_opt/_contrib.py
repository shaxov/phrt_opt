import numpy as np
from scipy import stats

from . import eig
from . import methods
from . import initializers
from .utils import randn_like


def retrieve_transmission_matrix(
    phases,
    intens,
    *,
    method="alternating_projections",
    initializer="random",
    max_attempts=10,
    verbose=0,
    phases_pinv=None,
    **kwargs,
):
    assert max_attempts >= 1
    p_value = 0.995
    
    # get number of columns
    _, m = intens.shape
    
    # make initializer
    if isinstance(initializer, str):
        initializer = initializers.get(initializer)
        if initializer.name() != "random":
            tol = kwargs.pop("power_method_tol", 1e-3)
            initializer = initializer(eig.PowerMethod(tol))
        else:
            initializer = initializer()

    if initializer.name() != "random":
        p_value = 0.999
            
    # get method
    method = methods.get(method)
    
    # initialize rows
    tm_0 = np.concatenate([initializer(phases, intens[:, [i]]) for i in range(m)], axis=-1)
    
    # first attempt to retrieve rows
    if phases_pinv is None:
        phases_pinv = np.linalg.pinv(phases)
    tm_k = np.concatenate([
        method(phases, intens[:, [i]], x0=tm_0[:, [i]], tm_pinv=phases_pinv, **kwargs)
        for i in range(m)
    ], axis=-1)
    
    # compute relative errors for each column
    r_err = np.linalg.norm(np.abs(phases @ tm_k) - intens, axis=0) / np.linalg.norm(intens, axis=0)
    
    # compute z-score
    z_score = stats.norm.ppf(p_value, loc=np.mean(r_err), scale=np.std(r_err))
    
    mask = r_err > z_score
    num_outliers = np.sum(mask)
    if verbose > 0:
        print(f"Attempt {1:2d}: number of outliers = {num_outliers:2d}")
    
    if not num_outliers:
        return tm_k
    
    # repeat attempts
    for attempt in range(1, max_attempts):
        
        tm_k_mask = np.concatenate([
            method(phases, intens[:, [i]], x0=randn_like(tm_0[:, [i]]), tm_pinv=phases_pinv, **kwargs)
            for i in np.nonzero(mask)[0]
        ], axis=-1)
        
        r_err_mask = (np.linalg.norm(np.abs(phases @ tm_k_mask) - intens[:, mask], axis=0) 
                      / np.linalg.norm(intens[:, mask], axis=0))
        
        improve_mask = r_err_mask < r_err[mask]
        mask[mask] = improve_mask
        
        tm_k[:, mask] = tm_k_mask[:, improve_mask]
        r_err[mask] = r_err_mask[improve_mask]
        
        mask = r_err > z_score
        num_outliers = np.sum(mask)
        if verbose > 0:
            print(f"Attempt {attempt + 1:2d}: number of outliers = {num_outliers:2d}")
            
        if not num_outliers:
            break
    return tm_k


def compute_camera_bias(phases, intens, btol=1e-4, verbose=0, **kwargs):
    iter_bias = np.inf
    total_bias = 0.
    
    phases_pinv = np.linalg.pinv(phases)
    
    iter_intens = intens
    it = 0
    while iter_bias > btol:
        tm = retrieve_transmission_matrix(
            phases, iter_intens, phases_pinv=phases_pinv, max_attempts=2, **kwargs)
        iter_model_intens = np.abs(phases @ tm)
        
        mat = np.vstack([np.ones_like(iter_model_intens.ravel()), iter_model_intens.ravel()]).T
        vec = iter_intens.ravel()
        iter_bias, _ = np.linalg.lstsq(mat, vec, rcond=None)[0]
        total_bias += iter_bias
        iter_intens = intens - total_bias
        if verbose > 0:
            print(f"Iteration {it + 1:2d}: camera bias = {total_bias:.4f}")
        it += 1
    return total_bias
