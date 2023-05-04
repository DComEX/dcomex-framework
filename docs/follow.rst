.. _follow:

Follow
======

This module provides a decorator follow that allows functions to be
"followed" in a call graph, as well as a function graphviz that
generates a call graph of the decorated functions. The decorator can
be used with an optional label for the wrapped function in the call
graph, and the generated graph can be written to a file-like object.


Summary
-------

.. autosummary::
   follow.follow
   follow.graphviz
   follow.loop

Functions
---------

.. autofunction:: follow.follow
.. autofunction:: follow.graphviz
.. autofunction:: follow.loop
