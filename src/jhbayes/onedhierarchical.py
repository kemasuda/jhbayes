__all__ = ["OnedHierarchical"]

#%%
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from jax.config import config
import pandas as pd
config.update('jax_enable_x64', True)

#%%
class OnedHierarchical():
    def __init__(self, samples_1d, bins_1d):
        self.xbins = bins_1d
        self.xmin, self.xmax, self.dx = bins_1d[0], bins_1d[-1], bins_1d[1] - bins_1d[0]
        self.Nbin = len(bins_1d) - 1
        self.xbins_center = 0.5 * (bins_1d[1:] + bins_1d[:-1])

        self.alpha_max = jnp.log(1./self.dx)
        self.alpha_mean = jnp.log(1./self.dx/self.Nbin)

        self.xidx = jnp.digitize(samples_1d, bins_1d) - 1
        self.L = jnp.array([jnp.sum(self.xidx == k, axis=1) for k in range(self.Nbin)]).T / jnp.shape(samples_1d)[1]

        self.xplot = np.linspace(self.xmin, self.xmax, 10000, endpoint=False)
        self.xidxplot = np.digitize(self.xplot, self.xbins)-1

    def gpmodel(self, sigma_prior=10., frac_data=None, test_prior=False):
        from jax.scipy.special import logsumexp
        import celerite2.jax
        from celerite2.jax import terms as jax_terms

        # hyperprior
        ones = jnp.ones(self.Nbin)
        alphas = numpyro.sample("alphas", dist.Normal(scale=sigma_prior*ones))
        alphas -= (jnp.log(self.dx) + logsumexp(alphas))
        priors = numpyro.deterministic("priors", jnp.exp(alphas))
        numpyro.deterministic("priors_sum", jnp.sum(priors) * self.dx)

        # GP prior on hyperprior
        lna = numpyro.sample("lna", dist.Uniform(low=-5, high=5))
        lnc = numpyro.sample("lnc", dist.Uniform(low=-3, high=3))
        mean = self.alpha_mean * ones
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.xbins_center)
        loglike_gp = numpyro.deterministic("loglike_gp", gp.log_likelihood(alphas))

        # likelihood
        if frac_data is not None:
            fracs = numpyro.sample("fracs", dist.Uniform(low=0*ones, high=ones))
            F = fracs**frac_data[:,None] * (1.-fracs)**(1.-frac_data[:,None])
        else:
            F = 1.

        # ignore data to test prior
        if test_prior:
            loglike_data = numpyro.deterministic("loglike_data", 0.)
        else:
            loglike_data = numpyro.deterministic("loglike_data", jnp.sum(jnp.log(jnp.dot(F*self.L, priors*self.dx))))

        numpyro.factor("loglike", loglike_data + loglike_gp)

    def run_hmc(self, target_accept_prob=0.9, num_warmup=1000, num_samples=1000, **kwargs):
        kernel = numpyro.infer.NUTS(self.gpmodel, target_accept_prob=target_accept_prob)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, **kwargs)
        mcmc.print_summary()
        return mcmc

    def summary_plots(self, mcmc, xlabel='$x$', ylabel='probability density', save=None):
        import corner
        import matplotlib.pyplot as plt
        x, xidx = self.xplot, self.xidxplot

        priors = np.array(mcmc.get_samples()["priors"])
        pmean = np.mean(priors, axis=0)
        p5, p95 = np.percentile(priors, [5, 95], axis=0)

        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(self.xmin, self.xmax)
        plt.plot(x, pmean[xidx], color="C0", label=u"mean")
        plt.fill_between(x, p5[xidx], p95[xidx], color="C0", alpha=0.1, label='90\%')
        #plt.plot(logpgrid, qmean, color="C1", label=u"posterior mean")
        #plt.fill_between(logpgrid, qmean-qstd, qmean+qstd, color="C1", alpha=0.1)
        plt.legend(loc="best")
        if save is not None:
            plt.savefig(save+"_models.png", dpi=200, bbox_inches="tight")
        plt.show()

        keys = ['lna', 'lnc']
        hyper = pd.DataFrame(data=dict(zip(keys, [mcmc.get_samples()[k] for k in keys])))
        labs = [k.replace("_", "") for k in keys]
        fig = corner.corner(hyper, labels=labs, show_titles="%.2f")
        if save is not None:
            fig.savefig(save+"_corner.png", dpi=200, bbox_inches="tight")
