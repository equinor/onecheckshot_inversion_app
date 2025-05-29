import numpy as np


def color_cells(val):
    """
    Styles the cell background color based on the value.

    Args:
      val: The value in the cell.

    Returns:
      The CSS style for the cell background.
    """
    if val == 0:
        color = "green"
    elif val == 1:
        color = "red"
    else:
        color = "white"
    return f"background-color: {color}"


def return_style_table(df):
    """
    Displays the DataFrame with colored cells using Streamlit.

    Args:
      df: The pandas DataFrame to display.
    """
    styled_df = df.style.applymap(color_cells)
    return styled_df


# function to apply decimation to a DataFrame
def decimate_dataframe(df, decimate_step):
    """Decimates a DataFrame by selecting every `decimate_step`-th row.

    Args:
        df: The input DataFrame.
        decimate_step: The decimation step size.

    Returns:
        The decimated DataFrame.
    """

    ibayes = np.arange(decimate_step - 1, len(df), decimate_step)
    ibayes[-1] = len(df) - 1  # Ensure the last row is included

    df_decimated = df.iloc[ibayes]
    return df_decimated
