import pymc3 as pm
import numpy as np
import theano
import theano.tensor as t
import matplotlib.pyplot as plt



# set the true values of the model parameters for creating the data
m = 3.5 # gradient of the line
c = 1.2 # y-intercept of the line

# set the "predictor variable"/abscissa
M = 100
xmin = 0.
xmax = 10.
stepsize = (xmax-xmin)/M
x = np.arange(xmin, xmax, stepsize)

# define the model function
def straight_line(x, m, c):
    """
    A straight line model: y = m*x + c
    
    Args:
        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """

    return m*x + c

# seed our random number generator, so we have reproducible data
np.random.seed(sum([ord(v) for v in 'samplers']))

# create the data - the model plus Gaussian noise
sigma = 0.5 # standard deviation of the noise
data = straight_line(x, m, c) + sigma*np.random.randn(M)

# plot the data
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(x, data, 'bo', alpha=0.5, label='data')
ax.plot(x, straight_line(x, m, c), 'r-', lw=2, label='model')
ax.legend()
ax.set_xlim([xmin, xmax])
ax.set_xlabel(r'$x$');
plt.show()



@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar],otypes=[t.dvector])
def model_fcn(mmodel, cmodel):
    print('mmode, cmodel: ', mmodel, cmodel)
    return mmodel*x + cmodel

with pm.Model() as model:
    # set prior parameters
    cmin = -10. # lower range of uniform distribution on c
    cmax = 10.  # upper range of uniform distribution on c
    
    mmu = 0.     # mean of Gaussian distribution on m
    msigma = 10. # standard deviation of Gaussian distribution on m
    
    # set priors for unknown parameters
    cmodel = pm.Uniform('c', lower=cmin, upper=cmax) # uniform prior on y-intercept
    mmodel = pm.Normal('m', mu=mmu, sd=msigma)       # Gaussian prior on gradient
    
    sigmamodel = sigma # set a single standard deviation
    
    # Expected value of outcome, aka "the model"
    # mu = mmodel*x + cmodel
    mu = model_fcn(mmodel, cmodel)

    # Gaussian likelihood (sampling distribution) of observations, "data"
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigmamodel, observed=data)


#>>>>>
# with model:
#     #start = pm.find_MAP()
#     #step =pm.NUTS()    
#     start ={'mmodel': 3.5, 'cmodel': 1.}
#     step = pm.Metropolis()

# with model:
#     trace = pm.sample(1000, step, start)

#<<<<    


#>>>>>
n_sample = 1000 # final number of samples
Ntune = 1000    # number of tuning samples



chains   = 4
burn     = int(0.1*n_sample)
thin     = 5

# perform sampling
# with model:
#     trace = pm.sample(n_sample, tune=Ntune, discard_tuned_samples=True); # perform sampling

with model:
    if chains == 1:
        kwargs['compute_convergence_checks'] = False
    trace = pm.sample(n_sample, chains=chains, init='adapt_diag', tune=Ntune)


#<<<<<

print( trace[mmodel].shape )
print( trace[cmodel].shape )
pm.traceplot(trace)
plt.show()