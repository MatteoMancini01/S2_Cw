import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers



def rad_tang_model(sections_ids, hole_ids, x_obs = None, y_obs = None):

    """
    Bayesian model for inferring the positions of holes in the Antikythera mechanism's calendar ring 
    using a radial-tangential error model.

    This function defines a probabilistic model where the hole positions are inferred based on a circular 
    arrangement with a given number of holes, radius, and section-specific transformations (translations and rotations).
    The observational errors are modeled using a Gaussian distribution with a covariance matrix that accounts 
    for radial and tangential uncertainties.

    Parameters
    ----------
    sections_ids : array-like (N,)
        An array specifying the section ID for each hole, used to apply section-specific transformations.
    
    hole_ids : array-like (N,)
        An array containing the hole indices, used to compute angular positions around the ring.

    x_obs : array-like (N,), optional
        The observed x-coordinates of the hole positions. Default is None.

    y_obs : array-like (N,), optional
        The observed y-coordinates of the hole positions. Default is None.

    Priors
    ------
    N : Uniform(330, 380)
        Prior on the total number of holes in the complete ring.

    r : Uniform(65, 85)
        Prior on the radius of the ring.

    sigma_t : LogUniform(0.001, 1)
        Prior on the standard deviation of the tangential error.

    sigma_r : LogUniform(0.001, 1)
        Prior on the standard deviation of the radial error.

    x_centre : Uniform(76, 86)
        Prior on the x-coordinate of the section center.

    y_centre : Uniform(130, 140)
        Prior on the y-coordinate of the section center.

    alpha_pred : Uniform(-3, -2)
        Prior on the angular misalignment for each section.

    Model
    -----
    - The hole locations are modeled as points on a circle of radius `r`, with section-specific 
      translations (`x_centre`, `y_centre`) and rotations (`alpha_pred`).
    - The covariance matrix `Sigma` models errors along radial and tangential directions.
    - The observed hole positions (`x_obs`, `y_obs`) are modeled as noisy measurements drawn from 
      a Multivariate Normal distribution centered at the predicted locations (`x_pred`, `y_pred`).

    Returns
    -------
    None
        This function defines a probabilistic model for inference using NumPyro.

    Notes
    -----
    - The model is implemented in NumPyro and requires JAX for computations.
    - The inference process should be run using MCMC sampling with NumPyro's inference algorithms.

    Example
    -------

    >>> import numpyro
    >>> from numpyro.infer import MCMC, NUTS

    >>> # Example data
    >>> sections_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> hole_ids = jnp.array([1, 2, 3, 4, 5, 6])
    >>> x_obs = jnp.array([75.1, 76.3, 78.5, 79.2, 80.1, 81.0])
    >>> y_obs = jnp.array([131.2, 132.8, 134.0, 135.1, 136.5, 137.0])

    >>> # Define the model
    >>> kernel = NUTS(rad_tang_model)
    >>> mcmc = MCMC(kernel, num_samples=1000, num_warmup=500)
    >>> mcmc.run(jax.random.PRNGKey(0), sections_ids, hole_ids, x_obs = x_obs, y_obs = y_obs)
    
    """

    # Defining our uniform priors
    N = numpyro.sample('N', dist.Uniform(330,380)) 
    r = numpyro.sample('r', dist.Uniform(65, 85))
    
    # Errors 
    sigma_t = numpyro.sample('sigma_t', dist.LogUniform(0.001, 1))
    sigma_r = numpyro.sample('sigma_r', dist.LogUniform(0.001,1))

    n_parameters = len(np.unique(sections_ids))

    with numpyro.plate('transform', n_parameters):
        x_centre = numpyro.sample('x_centre', dist.Uniform(70,90))
        y_centre = numpyro.sample('y_centre', dist.Uniform(125,145))
        alpha_pred = numpyro.sample('alpha_pred',dist.Uniform(-4, -1))

    # Defining model
    phase = 2*jnp.pi*(hole_ids - 1)/N  + alpha_pred[sections_ids]

    x_pred = r*jnp.cos(phase) + x_centre[sections_ids]
    y_pred = r*jnp.sin(phase) + y_centre[sections_ids]

    # Defining covariance matrix (Cartesian)
    Sigma = jnp.array(
        [jnp.array([[jnp.cos(phi), jnp.sin(phi)],[jnp.sin(phi),-jnp.cos(phi)]]).T @ jnp.diag(jnp.array([sigma_r**2, sigma_t**2])) @ jnp.array([[jnp.cos(phi),jnp.sin(phi)],[jnp.sin(phi),-jnp.cos(phi)]]) for phi in phase]
    )

    with numpyro.plate('data', len(hole_ids)):
        obs_data = None
        if x_obs is not None and y_obs is not None:
            obs_data = jnp.vstack((x_obs, y_obs)).T

        numpyro.sample('obs', dist.MultivariateNormal(
            loc=jnp.vstack((x_pred, y_pred)).T,
            covariance_matrix=Sigma
        ), obs=obs_data)



def isotropic_model(sections_ids, hole_ids, x_obs = None, y_obs = None):

    """
    Bayesian model for inferring hole positions in the Antikythera mechanism's calendar ring 
    using an isotropic Gaussian error model.

    This function defines a probabilistic model where hole positions are inferred assuming a 
    circular arrangement with a given number of holes, radius, and section-specific transformations.
    The observational errors are modeled using an isotropic Gaussian distribution with a 
    uniform variance in both radial and tangential directions.

    Parameters
    ----------
    sections_ids : array-like (N,)
        An array specifying the section ID for each hole, used to apply section-specific transformations.
    
    hole_ids : array-like (N,)
        An array containing the hole indices, used to compute angular positions around the ring.

    x_obs : array-like (N,), optional
        The observed x-coordinates of the hole positions. Default is None.

    y_obs : array-like (N,), optional
        The observed y-coordinates of the hole positions. Default is None.

    Priors
    ------
    N : Uniform(340, 370)
        Prior on the total number of holes in the complete ring.

    r : Uniform(74, 80)
        Prior on the radius of the ring.

    sigma : LogUniform(0.01, 1)
        Prior on the isotropic standard deviation of positional errors.

    x_centre : Uniform(76, 86)
        Prior on the x-coordinate of the section center.

    y_centre : Uniform(130, 140)
        Prior on the y-coordinate of the section center.

    alpha_pred : Uniform(-3, -2)
        Prior on the angular misalignment for each section.

    Model
    -----
    - The hole locations are modeled as points on a circle of radius `r`, with section-specific 
      translations (`x_centre`, `y_centre`) and rotations (`alpha_pred`).
    - The errors in measurements are modeled as isotropic Gaussian noise with covariance `Sigma`, 
      where `Sigma` is a diagonal matrix with variance `sigma^2` in both x and y directions.
    - The observed hole positions (`x_obs`, `y_obs`) are modeled as noisy measurements drawn from 
      a Multivariate Normal distribution centered at the predicted locations (`x_pred`, `y_pred`).

    Returns
    -------
    None
        This function defines a probabilistic model for inference using NumPyro.

    Notes
    -----
    - The model is implemented in NumPyro and requires JAX for computations.
    - The inference process should be run using MCMC sampling with NumPyro's inference algorithms.
    - Unlike the `rad_tang_model`, this model assumes equal uncertainty in both radial and tangential 
      directions, making it simpler but possibly less accurate.

    Example
    -------

    >>> import numpyro
    >>> from numpyro.infer import MCMC, NUTS

    >>> # Example data
    >>> sections_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> hole_ids = jnp.array([1, 2, 3, 4, 5, 6])
    >>> x_obs = jnp.array([75.1, 76.3, 78.5, 79.2, 80.1, 81.0])
    >>> y_obs = jnp.array([131.2, 132.8, 134.0, 135.1, 136.5, 137.0])

    >>> # Define the model
    >>> kernel = NUTS(isotropic_model)
    >>> mcmc = MCMC(kernel, num_samples=1000, num_warmup=500)
    >>> mcmc.run(jax.random.PRNGKey(0), sections_ids, hole_ids, x_obs = x_obs, y_obs = y_obs)
    ```
    """

    # Defining our uniform priors
    N = numpyro.sample('N', dist.Uniform(330,380)) 
    r = numpyro.sample('r', dist.Uniform(65, 85))
    
    # Errors 
    sigma = numpyro.sample('sigma', dist.LogUniform(0.001, 1))

    n_parameters = len(np.unique(sections_ids))

    with numpyro.plate('transform', n_parameters):
        x_centre = numpyro.sample('x_centre', dist.Uniform(70,90))
        y_centre = numpyro.sample('y_centre', dist.Uniform(125,145))
        alpha_pred = numpyro.sample('alpha_pred',dist.Uniform(-4, -1))

    # Defining model
    phase = 2*jnp.pi*(hole_ids - 1)/N  + alpha_pred[sections_ids]

    x_pred = r*jnp.cos(phase) + x_centre[sections_ids]
    y_pred = r*jnp.sin(phase) + y_centre[sections_ids]

    # Defining covariance matrix (Cartesian)
    Sigma = jnp.array([jnp.array([[sigma**2, 0],[0, sigma**2]]) for _ in phase])

    with numpyro.plate('data', len(hole_ids)):
        obs_data = None
        if x_obs is not None and y_obs is not None:
            obs_data = jnp.vstack((x_obs, y_obs)).T

        numpyro.sample('obs', dist.MultivariateNormal(
            loc=jnp.vstack((x_pred, y_pred)).T,
            covariance_matrix=Sigma
        ), obs=obs_data)
    



def hole_prediction(N, R, optim_parm, sec_mod):

    """
    Predicts the x and y coordinates of hole positions on the calendar ring 
    of the Antikythera mechanism, based on a circular model with section-specific 
    transformations.

    This function computes the expected Cartesian coordinates of each hole 
    assuming they are evenly spaced along a circle of radius R with N total holes. 
    Each section of the fractured ring is allowed its own translation and rotation 
    (phase shift), as provided in `optim_parm`.

    Parameters
    ----------
    N : int
        Total number of holes hypothesized in the original complete ring.
    
    R : float
        Estimated radius of the ring.

    optim_parm : tuple of arrays
        Tuple containing three 1D arrays representing the optimized transformation 
        parameters for each section:
        - optim_parm[0] : array of angular phase shifts (alpha_j) for each section.
        - optim_parm[1] : array of x-translation centers (x_0j) for each section.
        - optim_parm[2] : array of y-translation centers (y_0j) for each section.

    sec_mod : array-like
        An array mapping each hole index to a section ID (0-based). Should be of 
        length equal to the number of measured holes (typically 79).

    Returns
    -------
    jnp.ndarray
        A 2D array of shape (2, num_holes), where the first row contains the 
        predicted x-coordinates and the second row contains the predicted y-coordinates 
        of the holes.

    Notes
    -----
    - The function assumes 79 measured holes in total.
    - Hole angular positions are derived from a uniform angular spacing around the ring,
      offset by the section-specific phase shift.
    - Requires JAX for numerical operations (uses `jnp`).

    Example
    -------
    >>> xypred = hole_prediction(N=355, R=77.3, optim_parm=(alpha_j, x0_j, y0_j), sec_mod=section_map)
    >>> x_pred, y_pred = xypred[0], xypred[1]
    """

    x_pred = [] # x_ij
    y_pred = [] # y_ij
    
    # 79 measured holes
    for i in range(0, 79):

        phi_val = 2*jnp.pi*(i)/N + optim_parm[0][sec_mod[i]] # Computing phi_ij values

        x_val = R*jnp.cos(phi_val) + optim_parm[1][sec_mod[i]] # Computing x_ij values
        y_val = R*jnp.sin(phi_val) + optim_parm[2][sec_mod[i]] # Computing x_ij values

        x_pred.append(x_val)
        y_pred.append(y_val)
    
    return jnp.array([x_pred, y_pred])



def log_likelihood_rt_cartesian_jax(theta, data, N, sections_ids, hole_ids):

    """
    Compute the log-likelihood of the radial-tangential Gaussian error model for the
    Antikythera mechanism's calendar ring in Cartesian coordinates using JAX.

    This function calculates the log-likelihood assuming hole positions were originally
    arranged uniformly around a circle and that each fractured section may be translated
    and rotated. Errors in the observed positions are modeled with anisotropic Gaussian
    noise, separated into radial and tangential components.

    Parameters
    ----------
    theta : jnp.ndarray
        Parameter vector including [R, sigma_r, sigma_t, phases..., x_centers..., y_centers...]
        in that order. Used to compute the model predictions for each section.
    data : list of tuples
        Each element corresponds to a section, containing (x_obs, y_obs) arrays of observed
        positions for the holes in that section.
    N : int
        Hypothesized total number of holes in the original full ring (not just the observed ones).
    sections_ids : jnp.ndarray
        Array of section indices corresponding to each hole in the dataset.
    hole_ids : jnp.ndarray
        Array of hole indices (1-based) corresponding to each hole in the dataset.

    Returns
    -------
    float
        The total log-likelihood value computed under the radial-tangential Gaussian model.

    Notes
    -----
    - Uses JAX for automatic differentiation and vectorization.
    - Each hole's error is projected into radial and tangential directions for likelihood computation.
    - Assumes independent Gaussian errors for each direction with different variances.
    """


    R, sigma_r, sigma_t = theta[:3]
    phases, x_cents, y_cents = jnp.split(theta[3:], 3)

    inv_sig_r2 = 1.0/(sigma_r**2)
    inv_sig_t2 = 1.0/(sigma_t**2)

    loglike = 0.0
    npoints = 0

    for j in range(len(data)):
        x_obs, y_obs = data[j]
        n = len(x_obs)

        section_indices = (sections_ids == j)
        local_hole_ids = hole_ids[section_indices]

        phis = 2*jnp.pi*(local_hole_ids - 1)/N + phases[j]

        x_model = R*jnp.cos(phis) + x_cents[j]
        y_model = R*jnp.sin(phis) + y_cents[j]

        dx = x_obs - x_model
        dy = y_obs - y_model

        cphi = jnp.cos(phis)
        sphi = jnp.sin(phis)
        rp = dx*cphi + dy*sphi
        tp = dx*sphi - dy*cphi

        log_probs = -0.5*(rp**2*inv_sig_r2 + tp**2*inv_sig_t2)
        loglike += jnp.sum(log_probs)
        npoints += n

    norm = -npoints*jnp.log(2*jnp.pi*sigma_r*sigma_t)
    return norm + loglike


def log_likelihood_isotropic_cartesian_jax(theta, data, N, sections_ids, hole_ids):
    """
    Compute the log-likelihood of the isotropic Gaussian error model for the
    Antikythera mechanism's calendar ring in Cartesian coordinates using JAX.

    This function assumes holes were originally evenly spaced on a circle and
    accounts for section-specific translations and rotations. Positional errors
    are modeled as isotropic Gaussian noise (same standard deviation in both
    x and y directions). 

    Parameters
    ----------
    theta : jnp.ndarray
        Model parameters, ordered as:
        [R, sigma, phases..., x_centers..., y_centers...]
        - R: radius of the original ring
        - sigma: standard deviation of isotropic Gaussian noise
        - phases: angular misalignment per section
        - x_centers, y_centers: section-wise translation offsets
    data : list of tuples
        Observed hole positions for each section. Each tuple contains:
        (x_obs, y_obs): arrays of measured x and y coordinates.
    N : int
        Hypothesized total number of holes in the complete original ring.
    sections_ids : jnp.ndarray
        Array indicating which section each hole belongs to.
    hole_ids : jnp.ndarray
        Array of hole indices (1-based) corresponding to each hole in the dataset.

    Returns
    -------
    float
        Total log-likelihood under the isotropic Gaussian model. This value
        is suitable for optimisation or sampling-based inference.

    Notes
    -----
    - Uses JAX for efficient computation and automatic differentiation.
    - Assumes isotropic Gaussian noise (equal in radial and tangential directions).
    - Section-specific translations and rotations are modeled explicitly.
    """

    R, sigma = theta[:2]
    phases, x_cents, y_cents = jnp.split(theta[2:], 3)

    inv_sig2 = 1.0/(sigma**2)

    total_loglike = 0.0
    npoints = 0

    for j in range(len(data)):
        x_obs, y_obs = data[j]
        n = len(x_obs)

        section_indices = (sections_ids == j)
        local_hole_ids = hole_ids[section_indices]

        phis = 2 * jnp.pi * (local_hole_ids - 1) / N + phases[j]

        x_model = R * jnp.cos(phis) + x_cents[j]
        y_model = R * jnp.sin(phis) + y_cents[j]

        dx = x_obs - x_model
        dy = y_obs - y_model

        log_probs = -0.5 * (dx**2 + dy**2) * inv_sig2
        total_loglike += jnp.sum(log_probs)
        npoints += n

    norm = -npoints * jnp.log(2 * jnp.pi * sigma**2)
    return norm + total_loglike


def unpack_and_loglike_rt(theta, sections_ids, hole_ids, x_obs, y_obs):

    """
    Unpacks a flat parameter vector for the radial-tangential model and computes
    the log-likelihood of observed hole positions using NumPyro's tracing and 
    substitution tools.

    This function is designed to evaluate the log-likelihood of a specific parameter
    configuration (`theta`) for the radial-tangential error model of the Antikythera
    calendar ring. The parameters are assumed to follow a fixed structure, and the
    number of holes in the complete ring (N) is fixed to 354.0.

    Parameters
    ----------
    theta : array-like (length 21)
        Flattened vector of model parameters with the following structure:
        - theta[0]   : Radius `r` of the ring
        - theta[1]   : Radial standard deviation `sigma_r`
        - theta[2]   : Tangential standard deviation `sigma_t`
        - theta[3:9] : Angular offsets `alpha_pred` for each of 6 sections
        - theta[9:15]: x-centers `x_centre` for each section
        - theta[15:21]: y-centers `y_centre` for each section

    sections_ids : jnp.ndarray
        Section index for each observed hole, used to map holes to their corresponding
        transformation parameters.

    hole_ids : jnp.ndarray
        Hole indices (1-based) used to compute angular positions for model predictions.

    x_obs : jnp.ndarray
        Observed x-coordinates of hole positions.

    y_obs : jnp.ndarray
        Observed y-coordinates of hole positions.

    Returns
    -------
    float
        The total log-likelihood of the observed data under the specified parameters,
        based on the radial-tangential model using NumPyro.

    Notes
    -----
    - The model uses `N = 354` holes as a fixed assumption.
    - Utilises `numpyro.handlers.seed`, `substitute`, and `trace` to evaluate the 
      likelihood under a custom parameter setting.
    - Make sure that `rad_tang_model` is compatible with deterministic substitution.

    Example
    -------
    >>> theta = np.random.rand(21)
    >>> loglike = unpack_and_loglike_rt(theta, sections_ids, hole_ids, x_obs, y_obs)
    """


    # Unpack flat theta vector of length 21
    r = theta[0]
    sigma_r = theta[1]
    sigma_t = theta[2]
    alpha_pred = theta[3:9]
    x_centre = theta[9:15]
    y_centre = theta[15:21]

    # Set fixed N (or include in theta[21] if sampling it)
    N = 354.0

    # Create parameter dictionary
    params = {
        'N': N,
        'r': r,
        'sigma_r': sigma_r,
        'sigma_t': sigma_t,
        'alpha_pred': alpha_pred,
        'x_centre': x_centre,
        'y_centre': y_centre,
    }

    # Seeded and substituted model
    seeded_model = handlers.seed(rad_tang_model, rng_seed=0)
    sub_model = handlers.substitute(seeded_model, params)

    # Trace to extract log-likelihood
    trace = handlers.trace(sub_model).get_trace(sections_ids, hole_ids, x_obs=x_obs, y_obs=y_obs)
    loglike = trace["obs"]["fn"].log_prob(trace["obs"]["value"]).sum()

    return loglike



def unpack_and_loglike_is(theta, sections_ids, hole_ids, x_obs, y_obs):

    """
    Unpacks a flat parameter vector for the isotropic Gaussian error model and computes
    the log-likelihood of observed hole positions using NumPyro's tracing and substitution
    utilities.

    This function evaluates the likelihood of the observed hole data under a fixed set 
    of model parameters specified by a 1D array `theta`. The parameters define an isotropic 
    model where positional uncertainties are the same in both radial and tangential directions.
    The number of holes in the complete calendar ring is assumed to be fixed at N = 354.

    Parameters
    ----------
    theta : array-like (length 20)
        Flattened array of model parameters, in the following order:
        - theta[0]   : Radius `r` of the ring
        - theta[1]   : Isotropic standard deviation `sigma`
        - theta[2:8] : Angular offsets `alpha_pred` for each of the 6 sections
        - theta[8:14]: x-translation centers `x_centre` for each section
        - theta[14:20]: y-translation centers `y_centre` for each section

    sections_ids : jnp.ndarray
        Array of section indices for each observed hole, mapping holes to their corresponding
        ring fragment.

    hole_ids : jnp.ndarray
        Array of 1-based hole indices used to determine angular positions around the ring.

    x_obs : jnp.ndarray
        Observed x-coordinates of the hole positions.

    y_obs : jnp.ndarray
        Observed y-coordinates of the hole positions.

    Returns
    -------
    float
        The total log-likelihood of the observed data under the isotropic Gaussian model,
        computed using NumPyro's trace mechanism.

    Notes
    -----
    - The number of holes in the complete ring is fixed at N = 354.
    - Uses deterministic parameter substitution with `numpyro.handlers.substitute`.

    Example
    -------
    >>> theta = np.random.rand(20)
    >>> loglike = unpack_and_loglike_is(theta, sections_ids, hole_ids, x_obs, y_obs)
    """

    # Unpack flat theta vector of length 20
    r = theta[0]
    sigma = theta[1]
    alpha_pred = theta[2:8]
    x_centre = theta[8:14]
    y_centre = theta[14:20]

    # Set fixed N (or include in theta[21] if sampling it)
    N = 354.0

    # Create parameter dictionary
    params = {
        'N': N,
        'r': r,
        'sigma': sigma,
        'alpha_pred': alpha_pred,
        'x_centre': x_centre,
        'y_centre': y_centre,
    }

    # Seeded and substituted model
    seeded_model = handlers.seed(isotropic_model, rng_seed=0)
    sub_model = handlers.substitute(seeded_model, params)

    # Trace to extract log-likelihood
    trace = handlers.trace(sub_model).get_trace(sections_ids, hole_ids, x_obs=x_obs, y_obs=y_obs)
    loglike = trace["obs"]["fn"].log_prob(trace["obs"]["value"]).sum()

    return loglike