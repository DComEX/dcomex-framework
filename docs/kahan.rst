.. _kahan:

Kahan
=====

The kahan package provides efficient implementations of Kahan's
algorithms for numerical summation and other basic statistical
calculations. These algorithms are designed to reduce the loss of
precision that can occur when adding a large number of floating-point
values.

For more information on Kahan's algorithms and their implementation in
the kahan package, please see the package documentation.

References:

    Kahan, W. (1965). Pracniques: Further remarks on reducing
    truncation errors. Communications of the ACM, 8(1), 40-41.

Summary
-------

.. autosummary::
   kahan.cummean
   kahan.cumsum
   kahan.cumvariance
   kahan.mean
   kahan.sum

Functions
---------

.. autofunction:: kahan.cummean
.. autofunction:: kahan.cumsum
.. autofunction:: kahan.cumvariance
.. autofunction:: kahan.mean
.. autofunction:: kahan.sum
