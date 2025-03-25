import numpy as np

def model(x, y, R, phis, xcent, ycent, phase):

    ''' 
    Compute the radial and tangential errors between measured hole position and predicted hole locations
    in the fractured ring model.

    This function models the expected hole positions assuming an originally circular ring with N regularly spaced holes. 
    It then accounts for misaligment by applying transformation (xcent, ycent), in paper refered as $(x_{0j}, y_{0j})$, and a phase
    shift. Finally, it computes the errors between the measuered and predicted hole positions, projecting them into radial and 
    tangential components.

    Parameters:

    x : np.array
        Measured x-coordinates of hole positions for a specific fractured section. 
    y : np.array
        Measured y-coordinates of hole positions for a secific fracured section. 
    R : float
        Estimated radius of the original ring
    phis : np.array
        Angles of the holes in the unbroken ring, evenly spaced between 0 and 2pi
    xcent : float
        x-coordinate of the estimated center if the fractured section
    ycent : float
        y-coordinate of the estimated center of the fractured section
    phase : float
        Phase shift applied to align the hole positions.

    Returns:
    
    rp : np.array
        Radial errors (distance from the expected radius). 
    tp : np.array 
        Tangential errors (misaligment along the circumference of the ring). 

    Example:

    >>> x_measured = np.array([10.5, 11.2, 9.8])
    >>> y_measured = np.array([5.3, 4.9, 5.7])
    >>> R = 10
    >>> phis = np.linspace(0, 2*np.pi, len(x_measured))
    >>> xcent, ycent, phase = 0.5, -0.3, 0.1
    >>> rp, tp = model(x_measured, y_measured, R, phis, xcent, ycent, phase)

    >>> print(rp,tp)
    

    '''

    phi = phis + phase # Apply phase shift to the angles

    cphi = np.cos(phi) # Compute cosine values
    sphi = np.sin(phi) # Compute sine values

    # compute model points in x,y
    r_x = R*cphi # X-coordinates of prediceted holes
    r_y = R*sphi # Y-coordinates of predicted holes

    # shift data point to be around model x,y
    d_x = x - xcent 
    d_y = y - ycent

    # find error vector between data and model
    e_x = r_x - d_x
    e_y = r_y - d_y

    # project vector into radius and tangent
    rp = e_x*cphi + e_y*sphi # Radial projection
    tp = e_x*sphi - e_y*cphi # Tangential projection

    return rp, tp


def log_likelihood_rt(params, data, N):
    '''
    Compute log-likelihood for the radial-tangential Gaussian error model.

    Parameters:
    params (list): Model parameters [R, sigma_r, sigma_t, phase1, phase2, ..., xcent1, xcent2, ..., ycent1, ycent2, ...]
    data (list of tuples): Measured hole positions for each fractured section.
    N (int): Total number of holes in the original complete ring.

    Returns:
    float: Log-likelihood value.
    '''

    R, sigma_r, sigma_t = params[:3]
    phases, xcents, ycents = np.split(params[3:], 3)
    #x,y = data

    invsig_r = 1./(2*(sigma_r*sigma_r))
    invsig_t = 1./(2*(sigma_t*sigma_t))

    npoints = np.sum([len(dt) for dt in data])
    prefact = -npoints*np.log(2*np.pi*sigma_t*sigma_r)
    phis = 2*np.pi*np.arange(100)/N

    #k = np.arange(N)
    exp_likelihood = 0
    for i, sect in enumerate(data):
        x,y = sect

        # assume independent r, tangent
        rp, tp = model(x, y, R, phis[:len(x)], xcents[i], ycents[i], phases[i])

        exponent = -invsig_r*(rp**2) - invsig_t*(tp**2)

        exp_likelihood += np.sum(exponent)


    return prefact + exp_likelihood

def log_likelihood_isotropic(params, data, N):
    '''
    Compute log-likelihood for the isotropic Gaussian error model.

    Parameters:
    params (list): Model parameters [R, sigma, phase1, phase2, ..., xcent1, xcent2, ..., ycent1, ycent2, ...]
    data (list of tuples): Measured hole positions for each fractured section.
    N (int): Total number of holes in the original complete ring.

    Returns:
    float: Log-likelihood value.
    '''
    
    # Extract parameters
    R, sigma = params[:2]  # Only one sigma for isotropic model
    phases, xcents, ycents = np.split(params[2:], 3)

    # Compute inverse variance term
    invsig = 1./(2*sigma**2)

    # Compute number of points for normalization term
    npoints = np.sum([len(dt) for dt in data])
    prefact = -npoints*np.log(2*np.pi*sigma**2)

    # Generate equally spaced angles for the full ring
    phis = 2*np.pi*np.arange(100) / N

    # Compute log-likelihood
    exp_likelihood = 0
    for i, sect in enumerate(data):
        x, y = sect  # Measured hole positions in section i

        # Get predicted positions from the model
        m_x, m_y = model(x, y, R, phis[:len(x)], xcents[i], ycents[i], phases[i])

        # Isotropic Gaussian exponent
        exponent = -invsig*(m_x**2 + m_y**2)

        # Accumulate log-likelihood
        exp_likelihood += np.sum(exponent)

    return prefact + exp_likelihood



n_sec = 6 # number of sections
n_dim = 3+3*n_sec

# Define prior transform for 

# Radial-Tangential Model

def prior_transform_rt(u):
    """
    Transforms a point u in the unit cube [0,1]^d to the physical parameter space,
    matching the priors used in the radial-tangential model.

    Parameters:
        u (array-like): A 1D array of shape (ndim,) with values in [0, 1].

    Returns:
        theta (np.ndarray): Transformed parameters in the model's physical space.
    """
    
    ndim = n_dim
    theta = np.zeros(ndim)

    # Global parameters
    theta[0] = 65 + 20 * u[0]                    # R in [65, 85]
    theta[1] = 10**(-3 + 3 * u[1])               # sigma_r in [1e-3, 1] (log-uniform)
    theta[2] = 10**(-3 + 3 * u[2])               # sigma_t in [1e-3, 1] (log-uniform)

    # Section-wise parameters (phases, x-centres, y-centres)
    start = 3
    theta[start:start+n_sec] = -4 + 3 * u[start:start+n_sec]              # phase in [-4, -1]
    theta[start+n_sec:start+2*n_sec] = 70 + 20 * u[start+n_sec:start+2*n_sec]   # x_centre in [70, 90]
    theta[start+2*n_sec:start+3*n_sec] = 125 + 20 * u[start+2*n_sec:start+3*n_sec] # y_centre in [125, 145]

    return theta

# Isotropic Model

n_sec = 6 # number of sections
n_dim_iso = 2+3*n_sec

def prior_transform_isotropic(u):
    """
    Transforms a unit cube sample to physical parameter space for the isotropic model.

    Parameters:
        u (array-like): 1D array of shape (20,), values in [0, 1]

    Returns:
        theta (np.ndarray): Transformed parameters (20D)
    """
    
    ndim = 2 + 3 * n_sec
    theta = np.zeros(ndim)

    # Global parameters
    theta[0] = 65 + 20 * u[0]           # R in [65, 85]
    theta[1] = 10**(-3 + 3 * u[1])      # sigma in [1e-3, 1] (log-uniform)

    # Section-wise parameters
    start = 2
    theta[start:start+n_sec] = -4 + 3*u[start:start+n_sec]               # phase in [-4, -1]
    theta[start+n_sec:start+2*n_sec] = 70 + 20*u[start+n_sec:start+2*n_sec]     # x_centre
    theta[start+2*n_sec:start+3*n_sec] = 125 + 20*u[start+2*n_sec:start+3*n_sec] # y_centre

    return theta

def log_likelihood_rt_cartesian_np(theta, data, N, sections_ids, hole_ids):

    """
    Computes the log-likelihood for the radial-tangential Gaussian error model using NumPy.

    This function assumes that each section of the Antikythera mechanism's calendar ring
    may be translated and rotated independently, and that positional errors are anisotropic
    (different standard deviations in radial and tangential directions).

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector of length 21:
        - theta[0]   : Radius `R`
        - theta[1]   : Radial error standard deviation `sigma_r`
        - theta[2]   : Tangential error standard deviation `sigma_t`
        - theta[3:9] : Phase shifts `alpha_pred` for each section
        - theta[9:15]: x-centers `x_centre` for each section
        - theta[15:21]: y-centers `y_centre` for each section

    data : list of tuples
        Observed hole data for each section, where each element is (x_obs, y_obs),
        both np.ndarrays of equal length.

    N : int
        Hypothesized total number of holes in the full ring.

    sections_ids : np.ndarray
        Array mapping each observed hole to its corresponding section ID (0-based).

    hole_ids : np.ndarray
        Array of 1-based indices representing the position of each hole on the full ring.

    Returns
    -------
    float
        The total log-likelihood under the radial-tangential model with anisotropic noise.

    Notes
    -----
    - The function uses NumPy instead of JAX.
    - Assumes six sections in the ring with fixed parameter slicing.
    - Suitable for use in nested sampling or optimization.
    """
    R, sigma_r, sigma_t = theta[:3]
    phases, x_cents, y_cents = np.split(theta[3:], 3)

    inv_sig_r2 = 1.0/(sigma_r ** 2)
    inv_sig_t2 = 1.0/(sigma_t ** 2)

    loglike = 0.0
    npoints = 0

    for j, (x_obs, y_obs) in enumerate(data):
        n = len(x_obs)

        section_indices = (sections_ids == j)
        local_hole_ids = hole_ids[section_indices]

        phis = 2*np.pi*(local_hole_ids - 1)/N + phases[j]

        x_model = R*np.cos(phis) + x_cents[j]
        y_model = R*np.sin(phis) + y_cents[j]

        dx = x_obs - x_model
        dy = y_obs - y_model

        cphi = np.cos(phis)
        sphi = np.sin(phis)
        rp = dx*cphi + dy*sphi
        tp = dx*sphi - dy*cphi

        log_probs = -0.5*(rp**2*inv_sig_r2 + tp**2*inv_sig_t2)
        loglike += np.sum(log_probs)
        npoints += n

    norm = -npoints*np.log(2*np.pi*sigma_r*sigma_t)
    return norm + loglike


def log_likelihood_isotropic_cartesian_np(theta, data, N, sections_ids, hole_ids):

    """
    Computes the log-likelihood for the isotropic Gaussian error model using NumPy.

    This function models the observed hole positions as noisy measurements of an
    idealized circular ring with evenly spaced holes. Each section is translated
    and rotated independently. Measurement errors are assumed to be isotropic—
    i.e., having the same standard deviation in all directions.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector of length 20:
        - theta[0]   : Radius `R`
        - theta[1]   : Isotropic error standard deviation `sigma`
        - theta[2:8] : Phase shifts `alpha_pred` for each section
        - theta[8:14]: x-centers `x_centre` for each section
        - theta[14:20]: y-centers `y_centre` for each section

    data : list of tuples
        Observed hole data for each section, where each element is (x_obs, y_obs),
        both np.ndarrays of equal length.

    N : int
        Hypothesized total number of holes in the full ring.

    sections_ids : np.ndarray
        Array mapping each observed hole to its corresponding section ID (0-based).

    hole_ids : np.ndarray
        Array of 1-based indices representing the position of each hole on the full ring.

    Returns
    -------
    float
        The total log-likelihood under the isotropic error model.

    Notes
    -----
    - This version omits the normalization term in the likelihood by default.
    - Based on a simplified model where errors are directionally symmetric.
    - Useful for comparison with the more flexible radial-tangential model.
    """


    R, sigma = theta[:2]
    phases, x_cents, y_cents = np.split(theta[2:], 3)

    inv_sig2 = 1.0/(sigma ** 2)

    total_loglike = 0.0
    npoints = 0

    for j, (x_obs, y_obs) in enumerate(data):
        n = len(x_obs)

        section_indices = (sections_ids == j)
        local_hole_ids = hole_ids[section_indices]

        phis = 2*np.pi*(local_hole_ids - 1)/N + phases[j]

        x_model = R*np.cos(phis) + x_cents[j]
        y_model = R*np.sin(phis) + y_cents[j]

        dx = x_obs - x_model
        dy = y_obs - y_model

        log_probs = -0.5*(dx**2 + dy**2)*inv_sig2
        total_loglike += np.sum(log_probs)
        npoints += n

    norm = -npoints*np.log(2*np.pi*sigma**2)
    return norm + total_loglike

