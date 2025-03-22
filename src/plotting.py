import matplotlib.pyplot as plt
import numpy as np
import corner
import seaborn as sns

def plotting_comparison(observed_holes, predicted_holes, text_model):

    """
    Plots a comparison between measured (observed) and predicted hole positions 
    for different sections of the Antikythera mechanism.

    This function generates a scatter plot where the measured hole positions 
    are displayed with a semi-transparent color, while the predicted positions 
    are overlaid in a solid color. The section IDs are used to differentiate 
    points using distinct colors.

    Parameters
    ----------
    observed_holes : pandas.DataFrame
        A DataFrame containing observed (measured) hole positions with at least the 
        following columns:
        - "Mean(X)": The x-coordinates of the measured positions.
        - "Mean(Y)": The y-coordinates of the measured positions.
        - "Section ID": The section identifier for each hole.

    predicted_holes : pandas.DataFrame
        A DataFrame containing predicted hole positions with the same structure as 
        `observed_holes`, representing the model's inferred positions.

    text_model : str
        A string representing the name of the model being evaluated. It is used in 
        the plot title to indicate which model's predictions are being compared.

    Returns
    -------
    None
        Displays a scatter plot showing the comparison of measured vs predicted 
        hole positions.

    Notes
    -----
    - Different sections are represented with unique colors from the `tab10` colormap.
    - The predicted positions are labeled separately to distinguish them from the 
      measured data.
    - The y-axis inversion (commented) can be activated if needed for a better 
      visualization resembling a ring shape.

    Example
    -------

    >>> plotting_comparison(sub_data, pred_data_rt_HMC, 'Radial-Tangential Model')


    This will generate a scatter plot comparing observed and predicted hole positions 
    for the Radial-Tangential Model.
    """

    # Define colors using a colormap
    colors = plt.cm.get_cmap("tab10", len(observed_holes["Section ID"].unique()))

    # Plot
    plt.figure(figsize=(9, 6))

    # Loop through each section and plot automatically
    # Selecting different region based on Section ID
    for i, (section, data) in enumerate(observed_holes.groupby("Section ID")):
        plt.scatter(data["Mean(X)"], data["Mean(Y)"], s=20, alpha = 0.3, color=colors(i), label=f"Section ID = {section}")

    for i, (section, data) in enumerate(predicted_holes.groupby("Section ID")):
        plt.scatter(data["Mean(X)"], data["Mean(Y)"], s=20, color=colors(i), label=f"Section ID Predicted = {section}")

    # Add labels and title
    plt.xlabel('Mean(X)')
    plt.ylabel('Mean(Y)')
    plt.title(f'Measured vs Predicted Data ({text_model})')

    # Invert y-axis for better visualization (resemble ring shape)
    # plt.gca().invert_yaxis()

    # Add legend, adjust layout and grid
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Show plot
    plt.show()

def visualise_data(df_data):

    """
    Visualizes the hole locations from the given dataset, grouping them by section ID.

    This function generates a scatter plot where each section's holes are represented 
    using distinct colors. The function automatically assigns colors based on section 
    IDs and plots the hole positions on a 2D plane.

    Parameters
    ----------
    df_data : pandas.DataFrame
        A DataFrame containing hole position data with at least the following columns:
        - "Mean(X)": The x-coordinates of the hole positions.
        - "Mean(Y)": The y-coordinates of the hole positions.
        - "Section ID": The section identifier for each hole.

    Returns
    -------
    None
        Displays a scatter plot showing the spatial distribution of hole positions 
        grouped by section.

    Notes
    -----
    - Different sections are represented with unique colors from the `tab10` colormap.
    - The function automatically iterates over unique section IDs for visualization.
    - The y-axis inversion (commented) can be enabled if needed to better resemble 
      the actual structure of the ring.

    Example
    -------

    >>> visualise_data(df_hole_positions)


    This will generate a scatter plot showing the hole locations categorized by section.
    """

    # Define colors using a colormap
    colors = plt.cm.get_cmap("tab10", len(df_data["Section ID"].unique()))

    # Plot
    plt.figure(figsize=(9, 6))

    # Loop through each section and plot automatically
    # Selecting different region based on Section ID
    for i, (section, data) in enumerate(df_data.groupby("Section ID")):
        plt.scatter(data["Mean(X)"], data["Mean(Y)"], s=20, color=colors(i), label=f"Section ID = {section}")

    # Add labels and title
    plt.xlabel('Mean(X)')
    plt.ylabel('Mean(Y)')
    plt.title('Visualisation of Holes Location')

    # Invert y-axis for better visualization (resemble ring shape)
    #plt.gca().invert_yaxis()

    # Add legend, adjust layout and grid
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Show plot
    plt.show()


def corner_plots(posterior_for_isotropic, posterior_for_rad_tang):

    """
    Generates corner plots for posterior distributions of both the 
    isotropic and radial-tangential models.

    Parameters
    ----------
    posterior_for_isotropic : dict
        Posterior samples for the isotropic model.
    
    posterior_for_rad_tang : dict
        Posterior samples for the radial-tangential model.

    Returns
    -------
    None
        Displays corner plots comparing parameter distributions for both models.

    Example
    -------
    >>> corner_plots(posterior_samples_iso, posterior_samples_rt)
    """

    # Select the parameters to plot
    params_rt = ["N", "r", "sigma_r", "sigma_t"] 
    params_is = ["N", "r", "sigma"]

    # Convert posterior samples to a NumPy array for plotting
    samples_array_rt = np.column_stack([posterior_for_rad_tang[param] for param in params_rt])
    samples_array_is = np.column_stack([posterior_for_isotropic[param] for param in params_is])

    # Create a corner plot for the Radial-Tangential model
    figure_rt = corner.corner(samples_array_rt, 
                              labels=[r"$N$", r"$r$", r"$\sigma_r$", r"$\sigma_t$"], 
                              quantiles=[0.15, 0.5, 0.85],  
                              show_titles=True, 
                              title_kwargs={"fontsize": 14}, 
                              label_kwargs={"fontsize": 14}, 
                              bins=30, 
                              smooth=1)
    
    # Add a title
    plt.suptitle("Corner Plot for Radial-Tangential Model", fontsize=16, y=1.05)

    # Show the first plot
    plt.show()

    # Create a corner plot for the Isotropic model
    figure_is = corner.corner(samples_array_is, 
                              labels=[r"$N$", r"$r$", r"$\sigma$"], 
                              quantiles=[0.15, 0.5, 0.85],  
                              show_titles=True, 
                              title_kwargs={"fontsize": 14}, 
                              label_kwargs={"fontsize": 14}, 
                              bins=30, 
                              smooth=1)

    # Add a title
    plt.suptitle("Corner Plot for Isotropic Model", fontsize=16, y=1.05)

    # Show the second plot
    plt.show()


def joint_plot_sns(predictive_posteriror, hole_index, Data, title_ajustment):

    """
    Creates a 2D joint plot showing the posterior predictive distribution of a single hole 
    using seaborn's scatter and KDE contours.

    The plot includes:
    - Blue scatter of predicted positions from the posterior predictive distribution
    - KDE contour lines to represent density
    - Red dot for the observed hole position
    - Red 'x' for the predictive mean

    Parameters
    ----------
    predictive_posteriror : ndarray (num_samples, num_holes, 2)
        Posterior predictive samples of hole positions. Typically obtained from Predictive(...).
    
    hole_index : int
        The index of the hole to visualize.

    Data : pandas.DataFrame
        DataFrame containing observed x and y coordinates of the holes. Must include 'Mean(X)' and 'Mean(Y)'.

    title_ajustment : str
        A string indicating the model used (e.g., 'Radial-Tangential Model'), shown in the plot title.

    Returns
    -------
    None
        Displays the joint plot with KDE contours and annotations.

    Example
    -------
    >>> joint_plot_sns(posterior_predictive_samples_rt['obs'], hole_index=69, Data=sub_data, title_ajustment='Radial-Tangential Model')
    """

    x_obs = Data['Mean(X)'].to_numpy()
    y_obs = Data['Mean(Y)'].to_numpy()

    x_prediction = predictive_posteriror[:, hole_index, 0]
    y_prediction = predictive_posteriror[:, hole_index, 1]

    # Create the jointplot object
    g = sns.jointplot(
        x=x_prediction, 
        y=y_prediction, 
        kind='scatter', 
        s=10, 
        marginal_kws=dict(bins=60, fill=True), 
        color='blue', 
        alpha=0.6
    )

    # Add KDE contours
    sns.kdeplot(
        x=x_prediction, 
        y=y_prediction, 
        ax=g.ax_joint, 
        color='black', 
        alpha=0.3
    )

    # Add observed hole location, and predicted mean

    plt.scatter(x_obs[hole_index], y_obs[hole_index], color = 'red', s = 20, label = f"Hole {hole_index} Location")
    plt.scatter(x_prediction.mean(), y_prediction.mean(), color = "red", marker = 'x', label = f"Hole {hole_index} Predictive Mean")

    # Set axis labels
    g.set_axis_labels("X", "Y")

    # Add title to the whole figure, not the joint axes
    g.figure.suptitle(f"Posterior Predictive Distribution of Hole {hole_index} ({title_ajustment})", fontsize=14)

    # Adjust spacing so the title doesn't overlap
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.95)  # push plot down to make space for title

    plt.legend()

    plt.show()