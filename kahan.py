def cumvariance(a):
    """
    Cumulative mean and variance.

    Return an iterator over yielding pairs of cumulative mean and
    cumulative variance of an input sequence

    Parameters
    ----------
    a : iterable
        Input sequence

    Returns
    -------
    iterator
        An iterator that yields pairs of cumulative mean and cumulative variance.

    Examples
    --------
    >>> list(cumvariance([1, 7, 4]))
    [(1.0, 0.0), (4.0, 9.0), (4.0, 6.0)]

    """

    n = 0
    s = 0.0
    s2 = 0.0
    for e in a:
        n += 1
        y = e - s
        s += y / n
        s2 += y * (e - s)
        yield s, s2 / n


def cummean(a):
    """
    Cumulative mean.

    Return an iterator over yielding cumulative means of an input sequence

    Parameters
    ----------
    a : iterable
        Input sequence

    Returns
    -------
    iterator
        An iterator that yields cumulative means.

    Examples
    --------
    >>> list(cummean([1, 2, 3, 4]))
    [1.0, 1.5, 2.0, 2.5]

    """

    s = 0.0
    c = 0.0
    n = 0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
        n += 1
        yield s / n


def cumsum(a):
    """
    Cumulative sum.

    Return an iterator over yielding cumulative sums of an input sequence

    Parameters
    ----------
    a : iterable
        Input sequence

    Returns
    -------
    iterator
        An iterator that yields cumulative sums.

    Examples
    --------
    >>> list(cumsum([1, 2, 3, 4]))
    [1.0, 3.0, 6.0, 10.0]

    """

    s = 0.0
    c = 0.0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
        yield s


def sum(a):
    """
    Return the sum of iterable `a'.

    Parameters
    ----------
    a : iterable
        Input sequence

    Returns
    -------
    float
        The cumulative sum of the input sequence.

    Examples
    --------
    >>> sum([1, 2, 3, 4])
    10.0

    """

    s = 0.0
    c = 0.0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def mean(a):
    """
    Return the mean of iterable `a'.

    Return the cumulative mean of an input sequence

    Parameters
    ----------
    a : iterable
        Input sequence

    Returns
    -------
    float
        The cumulative mean of the input sequence.

    Examples
    --------
    >>> mean([1, 2, 3, 4])
    2.5

    """

    s = 0.0
    c = 0.0
    n = 0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
        n += 1
    return s / n
