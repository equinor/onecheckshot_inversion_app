def color_cells(val):
  """
  Styles the cell background color based on the value.

  Args:
    val: The value in the cell.

  Returns:
    The CSS style for the cell background.
  """
  if val == 0:
    color = 'green'
  elif val == 1:
    color = 'red'
  else:
    color = 'white'
  return f'background-color: {color}'

def return_style_table(df):
  """
  Displays the DataFrame with colored cells using Streamlit.

  Args:
    df: The pandas DataFrame to display.
  """
  styled_df = df.style.applymap(color_cells)
  return styled_df