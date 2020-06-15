def ttest_1samp(a, popmean, dim):
    """
    This is a two-sided test for the null hypothesis that the expected value
    (mean) of a sample of independent observations `a` is equal to the given
    population mean, `popmean`
    
    Inspired here: https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L3769-L3846
    
    Parameters
    ----------
    a : xarray
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    dim : string
        dimension along which to compute test
    
    Returns
    -------
    mean : xarray
        averaged sample along which dimension t-test was computed
    pvalue : xarray
        two-tailed p-value
    """
    n = a[dim].shape[0]
    df = n - 1
    a_mean = a.mean(dim)
    d = a_mean - popmean
    v = a.var(dim, ddof=1)
    denom = xrf.sqrt(v / float(n))

    t = d /denom
    prob = stats.distributions.t.sf(xrf.fabs(t), df) * 2
    prob_xa = xr.DataArray(prob, coords=a_mean.coords)
    return a_mean, prob_xa