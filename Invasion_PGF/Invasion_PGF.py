import numpy
import scipy.special
from odeintw import odeintw
from scipy import integrate

r''' Notes:
check license on odeintw")

change greek symbols for consistency with text modifications

there is opportunity for speedup in some functions when intermediate_values is false
'''

#
def _get_pts_(numpts,radius):
    r'''returns numpy array of numpts roots of unity on circle, scaled by given radius

    Arguments: 
    
        numpts : Integer giving the number of points

        radius : The radius

    Returns : 
        the solutions of z^nmpts = 1 in a numpy array
    '''
    return radius*numpy.exp(2*numpy.pi * 1j *numpy.linspace(0.,numpts-1.,numpts)/numpts)

def _get_coeff_(fxn_values, n, radius = 1):
    r'''Give function values at points on circle, calculate
    the nth coefficient of the function's expansion using Cauchy
    integral.

    p_n = [1/(2 pi i)] * int f(z)/z^n dz

    where integral is on circle of radius 'radius' in complex plane.
    
    Arguments :

        fxn_values : numpy array
            take numpts to be the length of fxn_values
            function values at locations radius e^{2 pi i k/numpts} for
            k = 0, 1, ..., numpts-1.

        n : We are looking for the coefficient of x^n

        radius : default to 1, but smaller may be more accurate.

    Returns : 
        p_n : predicted coefficient

    '''
    numpts = len(fxn_values)

    points = _get_pts_(numpts, radius)

    summation = sum(fxn_values/points**(n))

    integral  = summation/numpts

    return integral#numpy.real(integral)


def R0(offspring_PGF, dx = 10**(-10), central_diff = True):
    r'''Approximates R0

    Arguments : 

        offspring_PGF : function
            the PGF of the offspring distribution
        dx : float [default 10**(-10)]
            dx to use for numerical derivative
        central_diff : boolean (default True)
            if True then uses a central different approximation - usually much better

            if False : then approximates based on left hand side,
                       we probably only want this if function
                       doesn't converge for z>1

    Returns : 
        approximation to R0 by numerically estimating derivative of offspring_PGF.

    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.
        
        R0 = pgf.R0(mu)
        R0      #exact value is 1.5
        > 1.5000001241105565  
    '''
    if central_diff:
        #print('a')
        return (offspring_PGF(1+dx/2.)-offspring_PGF(1-dx/2.))/dx
    else:
        #print('b')
        return (offspring_PGF(1) - offspring_PGF(1-dx))/dx
    


def extinction_prob(offspring_PGF, Gen, intermediate_values = False):
    r'''
    Finds the probability of extinction by generation Gen
    [and in all intermediate generations if intermediate_values is True]

    Counting starts with generation 0.

    Convergence is generally quite quick, so a largish value of 
    Gen will give the overall extinction probability

    Arguments : 
        offspring_PGF : function
            The PGF of the offspring distribution.
        Gen : integer (non-negative)
            stop calculations with generation Gen 
        intermediate_values : boolean (default False)
            if True, return numpy array of [alpha_0, alpha_1, ..., alpha_Gen
               note that length of array is Gen+1
            if False, return alpha_Gen.
            

    Returns : 
        if intermediate_values is True:
            returns numpy array of [alpha_0, alpha_1, ..., alpha_Gen
               note that length of array is Gen+1
            if False, return alpha_Gen.

    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.
        
            
        alphas  = pgf.extinction_prob(mu, 10, intermediate_values = True)
        #The optional argument intermediate_values means that it gives everything
        #from generation 0 to 10 (inclusive)

        alphas
        > array([ 0.        ,  0.25      ,  0.33203125,  0.36972018,  0.38923784,
        0.39992896,  0.40595951,  0.40941651,  0.41141639,  0.4125794 ,
        0.41325779])

        alpha = pgf.extinction_prob(mu, 100)
        #calculates the extinction probability after generation 100.

        alpha
        > 0.41421356237309503
        
    '''
    alphas = []
    for g in range(Gen+1):
        if not alphas:
            alphas=[0]
        else:
            alphas.append(offspring_PGF(alphas[-1]))
    alphas = numpy.real(alphas) # removes numerical noise from imaginary part
    if intermediate_values:
        return numpy.array(alphas)
    else:
        return alphas[-1]
    

def active_infections(offspring_PGF, Gen, M=100, radius=1, numpts=1000, intermediate_values = False):
    r'''
    Calculates the probability of having 0, 1, ..., M-1 active infections
    in generation Gen.  [and in all intermediate generations if 
    intermediate_values is True]

    corresponds to coefficients for phi in the tutorial

    Arguments : 
        offspring_PGF : function
            The PGF of the offspring distribution.

        Gen : integer (non-negative)
            stop calculations with generation Gen 

        M : integer (default 100)
            returns probababilities of sizes from 0 to M-1

        radius : positive number (default 1)
            radius to use for the integral.  

        numpts : positive integer (default 1000)
            number of points on circle to use in calculating approximate coefficient
            needs to be bigger than M (much bigger for good accuracy)

        intermediate_values : boolean (default False)
            if True, return values for generations from 0 to Gen
            if False, just returns generations Gen

    Returns :
        if intermediate_values is True, return numpy array of numpy arrays.
            phis[g,n] is probability of n active infections in generation g.
            numpy array has g from 0 to Gen inclusive.
        if it is false, just returns numpy array
            phi[n] is probability of n active infections in generation Gen.


    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.
        
            
        Phi  = pgf.active_infections(mu, 10, M=5)
        #probabilities of 0, 1, 2, 3, or 4 active infections in generation 10
        Phi
        > array([ 0.41729954,  0.00502403,  0.00547124,  0.0061133 ,  0.00599666])
    '''
    if numpts<=M:
        print("warning numpts should be larger than M")
    ys = _get_pts_(numpts, radius)

    phis = []
    for g in range(Gen+1):
        if g==0:
            fxn_values = ys
        else:
            fxn_values = offspring_PGF(fxn_values)
        if intermediate_values or g==Gen:
            coeffs = []
            for n in range(M):
                coeffs.append(_get_coeff_(fxn_values, n, radius))
            phis.append(numpy.array(coeffs))
    phis = numpy.real(phis)  # removes numerical noise from imaginary part
    if intermediate_values:
        return numpy.array(phis)
    else:
        return phis[-1]
                            


def completed_infections(offspring_PGF, Gen, M=100, radius=1., numpts = 1000, intermediate_values = False):
    r'''

    Gives the probability of 0, 1, ..., M-1  completed infections 
    at generation Gen [and intermediate generations if 
    intermediate_values=True]

    Arguments : 
        offspring_PGF : function
            The PGF of the offspring distribution.
        Gen : integer (non-negative)
            stop calculations with generation Gen 
        M : integer (default 100)
            returns probababilities of sizes from 0 to M-1

        radius : positive number (default 1)
            radius to use for the integral.  

        numpts : positive integer (default 100)

        intermediate_values : boolean (default False)
            if True, return numpy array of arrays
               note that length of array is Gen+1
            if False, return just the array for generation Gen
                the array has the probability of n active infections from 0 to M-1

    Returns :
        if intermediate_values is True, return numpy array of numpy arrays.
            omegas[g,n] is probability of n completed infections in generation g.
            numpy array has g from 0 to Gen inclusive.
        if it is false, just returns numpy array
            omega[n] is probability of n active infections in generation Gen.


    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.
        
            
        omega  = pgf.completed_infections(mu, 3, M=10)
        #probabilities of 0, 1, 2, 3, ..., or 9 completed infections in generation 3
        omega
        > array([ -2.22044605e-18,   2.50000000e-01,   6.25000000e-02,
         7.81250000e-02,   9.76562500e-02,   1.21093750e-01,
         8.59375000e-02,   8.59375000e-02,   7.81250000e-02,
         6.25000000e-02])

        #note that the calculation is through a numerical integral, so the value
        #in first entry is 0 to numerical accuracy.
    '''
    if numpts<=M:
        print("warning numpts should be larger than M")

    zs = _get_pts_(numpts, radius)

    omegas = []
    for g in range(Gen+1):
        if g==0:
            fxn_values = numpy.ones(numpts)
        else:
            fxn_values = zs * offspring_PGF(fxn_values)

        if intermediate_values or g==Gen:
            coeffs = []

            for n in range(M):
                coeffs.append(_get_coeff_( fxn_values, n, radius))
            omegas.append(numpy.array(coeffs))
    omegas = numpy.real(omegas)# removes numerical noise from imaginary part
    if intermediate_values:
        return numpy.array(omegas)
    else:
        return omegas[-1]

def _get_pis_(Pis, M1, M2, intermediate_values, radius, threshold):
    Gen = len(Pis)-1
    pis = []
    for g, fxn_values in enumerate(Pis):
        if intermediate_values or g==Gen:
            coeffs = numpy.zeros((M1,M2))
            for n1 in range(M1):
                first_integral = numpy.array([_get_coeff_(row, n1, radius) for row in fxn_values])
                for n2 in range(M2):
                    full_integral = _get_coeff_(first_integral, n2, radius)
                    coeffs[n1,n2] = full_integral
            coeffs[coeffs<threshold]=0
            pis.append(coeffs)
    return numpy.array(pis)
    
def active_and_completed(offspring_PGF, Gen, M1=100, M2=100, radius=1, numpts = 1000, threshold = 10**(-10), intermediate_values = False):
    r'''

    gives probability of having 0, ...., M1-1 active infections and 
    0,..., M2-1 completed infections at generation Gen.  (joint distribution)

    [includes intermediate generations if intermediate_values is True]

    Arguments : 
        offspring_PGF : function
            The PGF of the offspring distribution.
        Gen : integer (non-negative)
            stop calculations with generation Gen 
        M1 : integer (default 100)
            consider 0, ..., M1-1 current infecteds
        M2 : integer (default 100)
            consider 0, ..., M2-1 completed infections
        radius : positive float (default 1)
            radius to use for integration
        numpts : integer (default 100)
            number of points to use for approximate integral.
        threshold : float [default 10**(-10)]
            any value below threshold is reported as 0.  Assumes that 
            calculation cannot be trusted at that size.
        intermediate_values : boolean (default False)
            if True, return numpy array of M1 x M2 arrays
               note that length of array is Gen+1
            if False, return M1xM2 array
                pi[n1,n2] = probability of n1 active and n2 completed infections

    Returns :
        if intermediate_values is True, return numpy array of M1 x M2 arrays
            note that length of array is Gen+1
        if False, return M1xM2 array
            pi[n1,n2] = probability of n1 active and n2 completed infections


    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.
        
            
        pi  = pgf.active_and_completed(mu, 3, M1=10, M2=20)
        #probability distribution of active and completed
        pi[2,4] #2 active 4 completed infections
        > 0.0097656250000000017

    '''
    if numpts<=M1 or numpts<=M2:
        print("warning numpts should be larger than M1 and M2")
    ys = _get_pts_(numpts, radius)
    zs = _get_pts_(numpts, radius)

    Pis = [] #pre integration version
    #pis = [] #post integration version
    for g in range(Gen+1):
        if g==0:
            fxn_values = numpy.array([ys for z in zs])
        else:
            fxn_values = numpy.array([z*offspring_PGF(row) for z, row in zip(zs,fxn_values)])
        Pis.append(fxn_values)

    pis = _get_pis_(Pis, M1, M2, intermediate_values, radius, threshold)
    #print(pis.shape)
    pis = numpy.real(pis) # removes numerical noise from imaginary part
    if intermediate_values:
        return numpy.array(pis)
    else:
        return pis[-1]


def final_sizes(PGF_function, M=100, numpts = 1000, radius=0.95):
    r'''
    
    Arguments:cts
        PGF_function
        M : returns probabilities of sizes 0, ..., M-1

    Returns: 
        sizes : numpy array of probabilities of sizes 0, ..., M-1
        
    :SAMPLE USE:

    ::

        import PGF_functions as pgf

        def mu(x):
            return (1 + x + x**2 + x**3)/4.

        sizes =  pgf.final_sizes(mu, 10)
        #probabilities of 0, 1, 2, 3, ..., or 9 total infections at end
        sizes
        > array([ 0.        ,  0.25      ,  0.0625    ,  0.03125   ,  0.01953125,
        > 0.01269531,  0.00878906,  0.00634766,  0.00471497,  0.003582  ])


    '''
    if numpts<=M:
        print("warning numpts should be larger than M")
    ys = _get_pts_(numpts, radius)
    mu_of_y = PGF_function(ys)
    fxn_values = numpy.ones(len(ys))
    coeffs = [0]


    for n in range(1,M):
        fxn_values = mu_of_y*fxn_values
        coeffs.append(_get_coeff_(fxn_values, n-1)/n)
    coeffs = numpy.real(coeffs)# removes numerical noise from imaginary part
    return numpy.array(coeffs)


#for the integration routines below, it might be nicer to use
#integrate.ode, but I'm more comfortable with odeint, so I'll
#stick to it.  I do not expect any performance issues to be a problem.

def _mu_hat_(beta, gamma, y, z):
    return (beta*y**2 + gamma*z)/(beta+gamma)
    
def _dalpha_dt_(X, t, beta, gamma):
    alpha = X[0]
    return [(beta+gamma)*(_mu_hat_(beta, gamma, alpha, 1)-alpha)]
    
    
def cts_time_extinction_prob(beta, gamma, T=None, intermediate_values = False, numvals = 11):
    r'''
    Gives probability of extinction by time T or at times in interval [0,T]
    for continuous-time model.
    
    Arguments : 
        
        beta : float
            transmission rate
        gamma : float
            recovery rate
        T : float (default None)
            stop time (if None, then just gives final extinction probability)
        intermediate_values : boolean (default False)
            irrelevant if T is None
            tells whether to return intermediate values 
            in [0,T]
        numvals : int (default 11)
            number of values in [0, T] inclusive to return.

    Returns:
        
        if T is None, returns 
            alpha,  a float
        
        if T is not None and intermediate_values is False : 
            alpha(T)
            
        if T is not None and intermediate_values is True : 
            numpy array of [alpha(0), alpha(T/numvals), alpha(2T/numvals), ...,
                            alpha(T)]
    '''
    if T is None:
        return min(1, gamma/beta)
    else:
        times = numpy.linspace(0, T, numvals)
        alpha0 = [0j]
        alphas = odeintw(_dalpha_dt_, alpha0, times, args = (beta, gamma))
        alphas = numpy.real(alphas)#removes numerical noise from imaginary part
        alphas = numpy.transpose(alphas)
        if intermediate_values:
            return (times, alphas)
        else:
            return alphas[-1]

def _dPhi_dt_(X, t, beta, gamma):
    phi = X
    return  (beta+gamma)*(_mu_hat_(beta, gamma, phi,1) - phi)
    
def cts_time_active_infections(beta, gamma, T, M=100, radius=1, numpts=1000, 
                                intermediate_values = False, numvals = 11):
    r'''
    Gives probability of having 0, ..., M-1 active infections at time T or at 
    times in interval [0,T] for continuous-time model.
    
    Arguments : 
        
        beta : float
            transmission rate

        gamma : float
            recovery rate

        T : float
            stop time 

        M : integer (default 100)
            returns probababilities of sizes from 0 to M-1

        radius : positive number (default 1)
            radius to use for the integral.  

        numpts : positive integer (default 100)
            number of points on circle to use in calculating approximate coefficient

        intermediate_values : boolean (default False)
            irrelevant if T is None
            tells whether to return intermediate values 
            in [0,T]

        numvals : int (default 11)
            number of values in [0, T] inclusive to return.

    Returns:
        
        if intermediate_values is False : 
            numpy array phi, where phi[n] is probability of n infections at time T.
            
        if intermediate_values is True : 
            (ts, numvals x M numpy array)
            ts[i] is ith time and where phi[i, n] is probability of n active
            infections at ith time.
    
    '''
    if numpts<=M:
        print("warning numpts should be larger than M")
    
    times = numpy.linspace(0, T, numvals)
    ys = _get_pts_(numpts, radius)
    Phis0 = ys+0j #just to make sure complex
    
    Phis = odeintw(_dPhi_dt_, Phis0, times, args = (beta, gamma))
    
    if intermediate_values:
        phis = []
        for fxn_values in Phis:
            coeffs = []
            for n in range(M):
                coeffs.append(_get_coeff_(fxn_values, n, radius))
            phis.append(numpy.array(coeffs))
        phis = numpy.array(phis)
        phis = numpy.real(phis) #remove numerical noise
        return (times, phis)
    else:
        fxn_values = Phis[-1]
        coeffs = []
        for n in range(M):
            coeffs.append(_get_coeff_(fxn_values, n, radius))
        coeffs = numpy.array(coeffs)
        coeffs = numpy.real(coeffs)  #remove numerical noise
        return coeffs
    
def _dOmega_dt_(X, t, beta, gamma, z):
    Omega = X
    return (beta+gamma)*(_mu_hat_(beta, gamma, Omega,  z) - Omega)
    
def cts_time_completed_infections(beta, gamma, T, M=100, radius=1., 
                                    numpts = 1000, 
                                    intermediate_values = False,
                                    numvals = 11):
    r'''
    Gives probability of having 0, ...., M-1 completed infections at time T.
    Arguments : 
        
        beta : float
            transmission rate

        gamma : float
            recovery rate

        T : float
            stop time 

        M : integer (default 100)
            consider 0, ..., M-1 current infecteds

        radius : positive number (default 1)
            radius to use for the integral.  

        numpts : positive integer (default 100)
            number of points on circle to use in calculating approximate coefficient

        threshold : float [default 10**(-10)]
            any value below threshold is reported as 0.  Assumes that 
            calculation cannot be trusted at that size.

        intermediate_values : boolean (default False)
            irrelevant if T is None
            tells whether to return intermediate values 
            in [0,T]

        numvals : int (default 11)
            number of values in [0, T] inclusive to return.

    Returns:
        
        if intermediate_values is False : 
            numpy array omega, where omega[n] is probability of n completed 
            infections at time T.
            
        if intermediate_values is True : 
            (ts, numvals x M numpy array)
            ts[i] is ith time and where Omega[i, n] is probability of n 
            completed infections at ith time.
    '''
    if numpts<=M:
        print("warning numpts should be larger than M")

    times = numpy.linspace(0, T, numvals)

    zs = _get_pts_(numpts, radius)

    Omega0 = numpy.ones(numpts)+0j #array of ones (but complex)
    
    Omegas = odeintw(_dOmega_dt_, Omega0, times, 
                                args = (beta, gamma, zs))

    if intermediate_values:
        omegas = []
        for fxn_values in Omegas:
            coeffs = []
            for n in range(M):
                integral = _get_coeff_(fxn_values, n, radius)
                coeffs.append(integral)
            omegas.append(coeffs)
        omegas = numpy.array(omegas)
        omegas = numpy.real(omegas)
        return (times, omegas)
    else:
        fxn_values = Omegas[-1]
        coeffs = []
        for n in range(M):
            integral = _get_coeff_(fxn_values, n, radius)
            coeffs.append(integral)
        coeffs = numpy.array(coeffs)
        coeffs = numpy.real(coeffs)
        return coeffs


#Danger of Pi and z not having appropriate orientation.  Check with M1 != M2
def _dPi_dt_(X, t, beta, gamma, zs):
    Pi = X
    #print(beta,gamma)
    #print(zs)
    returnarray = (beta + gamma)*(_mu_hat_(beta, gamma, Pi, zs)-Pi)
    return returnarray

def cts_time_active_and_completed(beta, gamma, T, M1=100, M2=100, radius=1, 
                                    numpts = 1000, threshold = 10**(-10), 
                                    intermediate_values = False, 
                                    numvals = 11):
    r'''
    Gives probability of having 0, ...., M1-1 active infections and 
    0,..., M2-1 completed infections at time T.  (joint distribution)

    Arguments : 
        
        beta : float
            transmission rate

        gamma : float
            recovery rate

        T : float
            stop time 
        M1 : integer (default 100)
            consider 0, ..., M1-1 current infecteds
        M2 : integer (default 100)
            consider 0, ..., M2-1 completed infections

        radius : positive number (default 1)
            radius to use for the integral.  

        numpts : positive integer (default 100)
            number of points on circle to use in calculating approximate coefficient

        threshold : float [default 10**(-10)]
            any value below threshold is reported as 0.  Assumes that 
            calculation cannot be trusted at that size.

        intermediate_values : boolean (default False)
            irrelevant if T is None
            tells whether to return intermediate values 
            in [0,T]

        numvals : int (default 11)
            number of values in [0, T] inclusive to return.

    Returns:
        
        if intermediate_values is False : 
            numpy array pi, where pi[n1,n2] is probability of n1 active and 
            n2 completed infections at time T.
            
        if intermediate_values is True : 
            (ts, numvals x M1xM2 numpy array)
            ts[i] is ith time and where pi[i, n1,n2] is probability of n1 active
            infections and n2 completed at ith time.
    '''
    if numpts<=M1 or numpts<=M2:
        print("warning numpts should be (quite a bit) larger than M1 and M2")
    times = numpy.linspace(0, T, numvals)

    ys = _get_pts_(numpts, radius)+0j
    zs = _get_pts_(numpts, radius)+0j
    Z = numpy.transpose(numpy.array([zs for y in ys]))

    Pis0 = numpy.array([ys for z in zs])
    Pis = odeintw(_dPi_dt_, Pis0, times, args = (beta, gamma, Z))
    
    pis = _get_pis_(Pis, M1, M2, intermediate_values, radius, threshold)
    pis = numpy.real(pis)
    if intermediate_values:
        return (times, pis)
    else:
        return pis[-1]
    
    
        
def cts_time_final_sizes(beta, gamma, M=100):
    r'''No contour integration needed.'''
    coeffs = [0] #j=0
    base = (beta+gamma)/beta
    betagammafactor = beta*gamma/(beta+gamma)**2
    base = base*betagammafactor
    coeffs.append(base*1/1)  #j=1 needs to be handled since 0 choose 0 = 1.
    for j in range(2,M):
        base = base * betagammafactor* (2*j-2)*(2*j-3)/(j-1)**2
        coeffs.append(base/j)
        #prob = beta**(j-1)*gamma**j *  scipy.special.binom(2*j-2,j-1) /j
        #coeffs.append(prob)
    if M<3:
        coeffs = coeffs[:M]
    coeffs = numpy.array(coeffs)
    coeffs = numpy.real(coeffs)
    return coeffs
