import matplotlib.pyplot as plt
import pandas as pd

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