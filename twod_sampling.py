import numpy as np
from scipy.interpolate import LinearNDInterpolator
import emcee


def sample2D(X,Y,d2P_dXdY,N,method='simple'):
    """
    Extract samples from a 2D probability distribution.
    
    Parameters:
    - X: x-axis grid over which the probability has been evaluated (1D numpy array of length n or 2D array of shape [m,n])
    - Y: y-axis grid over which the probability has been evaluated (1D numpy array of length m or 2D array of shape [m,n])
    - d2P_dXdY: 2D probability density (2D array of shape [m,n])
    - N: number of samples desired (int)
    
    Keywords:
    - method: either 'MCMC' or 'simple'
    
    Returns: X_i, Y_i
    - X_i, Y_i: 1D arrays of length N containing samples that follow the given 2D probability distribution
    
    """
    
    # find shape
    m,n = d2P_dXdY.shape
    
    # if X, Y is not a meshgrid, make one
    if len(X.shape)==1 or len(Y.shape)==1:
        x,y = np.meshgrid(X,Y)
    else:
        x = X
        y = Y
    
    z = d2P_dXdY
    
    # create interpolator
    I = LinearNDInterpolator(list(zip(x.ravel(),y.ravel())),np.log(z.ravel()),fill_value=-np.inf) # linear interpolation of the logarithm of the probability density
    
    
    """
    # test the interpolation to check if everything worked
    xx,yy = np.meshgrid(np.linspace(x.min(),x.max(),50),np.linspace(y.min(),y.max(),51))
    
    logzz = I(xx.ravel(),yy.ravel()).reshape(xx.shape)
    
    plt.contour(x,y,np.log(z),colors='k',linestyles='-',label='original')
    plt.contour(xx,yy,logzz,colors='r',linestyles='--',label='interpolated')
    
    plt.legend()
    plt.show()
    """
    
    if method=='MCMC':
        
        # define logprob function to be used in emcee
        loglike = lambda coords: I(coords[0],coords[1])
        
        # setup emcee sampler
        ndim = 2
        nwalkers = 4*ndim
        nsteps = int(1e7)
        
        # uniform initial conditions
        x0s = np.random.uniform(x.min(),x.max(),nwalkers)
        y0s = np.random.uniform(y.min(),y.max(),nwalkers)
        
        p0s = np.vstack([x0s,y0s]).T
        
    
        print(p0s.shape)
        
        # instantiate sampler
        sampler = emcee.EnsembleSampler(nwalkers,ndim,loglike)
        
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(nsteps)
        
        # This will be useful to testing convergence
        old_tau = np.inf
        
        for sample in sampler.sample(p0s, iterations=nsteps, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
        
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
        
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.1)
            if converged and (len(sampler.flatchain)>(2*N)): # require the chain to be at least 2*N long
                break
            old_tau = tau
        
        samples = sampler.flatchain[sampler.flatchain.shape[0]-int(N):,:]
    
    else:
        
        x0s = np.random.uniform(x.min(),x.max(),N*10)
        y0s = np.random.uniform(y.min(),y.max(),N*10)
        z0s = np.random.uniform(0.,z.max(),N*10)
        
        good = I(x0s,y0s)>=np.log(z0s)
        
        samples = np.vstack([x0s[good],y0s[good]]).T
    
    return samples
    

if __name__=='__main__':
    """
    Auto-test: generate samples from a given 2D PDF
    """
    
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    
    # contruct the 2D PDF
    x = np.linspace(-1,1.,100)
    y = np.linspace(-2.,2.,99)
    X,Y = np.meshgrid(x,y)
    P = (1.-np.abs(X))*np.exp(-Y**2) # PDF analytical form
    
    # normalize it
    P = P/np.trapz(np.trapz(P,X,axis=1),y,axis=0)
    
    # sample it
    samples = sample2D(X,Y,P,30000)
    
    print(samples.shape)
    
    # construct a 2D Gaussian Kernel Density estimator
    kde = gaussian_kde(samples.T)
    
    # plot the estimated density contours over the original ones
    plt.subplot(223)
    plt.contour(X,Y,P,label='original')
    plt.contour(X,Y,kde([X.ravel(),Y.ravel()]).reshape(X.shape),linestyles='--',label='reconstructed')
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    # plt.colorbar(label='Probability density contour')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    
    # do the same for the 1D marginalized distributions
    plt.subplot(221)
    
    Px = np.trapz(P,Y,axis=0)
    kdex = gaussian_kde(samples[:,0])
    
    plt.plot(x,Px,'-k')
    plt.plot(x,kdex(x),'--r')
    
    plt.tick_params(which='both',direction='in',top=True,right=True)
    
    plt.tick_params(which='both',direction='in',top=False,bottom=True,left=False,right=False,labelbottom=False,labelleft=False)
    
    plt.subplot(224)
    
    Py = np.trapz(P,X,axis=1)
    kdey = gaussian_kde(samples[:,1])
    
    plt.plot(y,Py,'-k')
    plt.plot(y,kdey(y),'--r')
    
    plt.tick_params(which='both',direction='in',top=False,bottom=True,left=False,right=False,labelbottom=True,labelleft=False)
    
    plt.xlabel('Y')
    
    plt.subplot(222)
    
    plt.plot([0],[0],'-k',label='original')
    plt.plot([0],[0],'--r',label='reconstructed')
    
    plt.tick_params(which='both',top=False,bottom=False,left=False,right=False,labelbottom=False,labelleft=False)
    
    plt.legend()
    
    plt.gca().set_axis_off()
    
    plt.show()
