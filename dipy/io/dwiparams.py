''' Process diffusion imaging parameters

* ``q`` is a vector in Q space
* ``b`` is a b value
* ``g`` is the unit vector along the direction of q (the gradient
  direction) 

Thus:

   b = |q|

   g = q / |q|

(|q| is the Euclidean norm of ``q``)

The B matrix ``B`` is a symmetric positive semi-definite matrix.  If
``q_est`` is the closest q vector equivalent to the B matrix, then:

   B ~ (q_est . q_est.T) / |q_est|

'''
import numpy as np
import numpy.linalg as npl


def B2q(B):
    ''' Estimate q vector from input B matrix `B`

    We assume the input `B` is symmetric positive definite.  If not,
    then you will get the result as for the lower triagular part of `B`.

    Because the solution is a square root, the sign of the returned
    vector is arbitrary.  We set the vector to have a positive x
    component by convention.

    Parameters
    ----------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.
       
    Returns
    -------
    q : (3,) vector
       Estimated q vector from B matrix `B`
    '''
    B = np.asarray(B)
    w, v = npl.eigh(B)
    tol = np.abs(w.max() * np.finfo(w.dtype).eps)
    non_trivial = np.abs(w) > tol
    if np.any(w[non_trivial] < 0):
        raise ValueError('B not positive semi-definite')
    inds = np.argsort(w)
    max_ind = inds[-1]
    vector = v[:,max_ind]
    # because the factor is a sqrt, the sign of the vector is arbitrary.
    # We arbitrarily set it to have a positve x value.
    if vector[0] < 0:
        vector *= -1
    return vector * w[max_ind]