import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist




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

    '''  
    Function to compute prediction of hole location on the xy-plane.

    Parameters:
    
    N : Int
        Number of predicted holes
    
    R : Int
        Predicted radius of the original circle

    optim_parm : 
        Predicted parameters (x_0j, y_0j, alpha_j)

    sec_mod : 
        Model's sections (0,1,2,3,4,5)

    Returns:
        np.array([x_pred, y_pred]), array of predicted x and y values.

    '''

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