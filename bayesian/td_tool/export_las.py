import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import base64
from io import BytesIO
import segyio
import lasio
import pandas as pd

def resample_log(zz, val, zz_out):
    ff = interp1d(zz, val, kind="next", bounds_error=False, fill_value=np.nan)
    return ff(zz_out)


def interpolate_with_extrapolation_and_null_check(A, B):
    """
    Creates a linear interpolation function that calculates B from A 
    using interp1d, extrapolates to values out of range, 
    and returns Null if B values are below 0.

    Args:
      A: numpy array of independent variable values.
      B: numpy array of dependent variable values.

    Returns:
      A function that takes an array of A values as input 
      and returns the interpolated B values.
    """

    # Create the interpolation function with extrapolation
    f = interp1d(A, B, kind='linear', fill_value='extrapolate') 

    def interpolate_and_check(A_new):
        """
        Interpolates B for new A values and handles null cases.

        Args:
          A_new: numpy array of new A values.

        Returns:
          A numpy array of interpolated B values, 
          or Null if any interpolated B value is below 0.
        """
        B_new = f(A_new)
        if np.any(B_new < 0):
            return None
        else:
            return B_new

    return interpolate_and_check

def to_las(depth_path, uwi,output_file, depth_in,vp_input,vp_ext,vp_output, depth_step, depth_export,answer_depth_convention,md_interp=None,depth_in_tvdss=None, null_value=-999.25):

    print("HERE",len(pd.DataFrame(depth_in, depth_in_tvdss)[1:]),len(pd.DataFrame(depth_in_tvdss, depth_in)) )
    
    
    # create output depth curve
    if depth_step is not False:
        depth = np.arange(depth_in[0], depth_in[-1] + depth_step, depth_step)
        depth = depth[depth <= depth_in[-1]]
        if depth_path == "MD":
            md_to_tvdss = interp1d(depth_in, depth_in_tvdss, kind='linear', fill_value=np.nan, bounds_error=False)
            depth_tvdss = md_to_tvdss(depth)
    else:
        depth = depth_in


    # initiate LAS-file
    las = lasio.LASFile()
    # header values
    las.well.WELL.value = uwi

    las.well["ELEV"] = lasio.HeaderItem(
        "ELEV", unit="m", value="0", descr="SURFACE ELEVATION"
    )

    # values
    log_name = ["vp_input", "vp_ext","vp_output"]
    log_unit = ["m/s", "m/s", "m/s"]
    log_descr = [
        "Vp_Input",
        "Vp_Extended",
        "Vp_Output_Bayesian"
    ]
    nlog = len(log_name)

    # add depth curves to LAS


    #las.add_curve("TVD", depth, unit="m", descr="True Vertical Depth KB")
    #las.add_curve("TVDMSL", depth, unit="m", descr="True Vertical Depth Mean Sea Level")

    # loop through data columns and add
    # las.add_curve('VP', resample_log(depth_in, vp, depth), unit='m/s', descr = 'P-velocity')
    dict_data = dict()
    for ii in np.arange(nlog):

        # get data
        val_in = eval(log_name[ii])

        # check if log is empty
        if len(val_in) > 0:

            # resample
            val = resample_log(depth_in, val_in, depth)

            # replace nan's
            val = np.nan_to_num(
                val, copy=True, nan=null_value, posinf=null_value, neginf=null_value
            )
            print(val)
            # add to las
            dict_data[log_name[ii].upper()] = [val, log_unit[ii], log_descr[ii]]
    print(answer_depth_convention)
    if answer_depth_convention == "Depth values above MSL are positive, while depth values below MSL are negative.":
        depth = -depth
        depth_tvdss = -depth_tvdss

    else:
        pass

    if depth_path == "MD":
        las.append_curve("DEPTH", depth, unit="m", descr="Measured Depth")
        las.append_curve("DEPTH_TVDSS", depth_tvdss, unit="m", descr="True Vertical Depth SubSea")
    elif depth_path == "TVDSS":
        las.append_curve("DEPTH", depth, unit="m", descr="True Vertical Depth SubSea")
            
    for log_curve in dict_data.keys():
            las.append_curve(log_curve, dict_data[log_curve][0], unit=dict_data[log_curve][1], descr=dict_data[log_curve][2])
    
    new_dict_data = {key: value[0] for key, value in dict_data.items()}
    df_output = pd.DataFrame(new_dict_data)
    df_output.insert(loc=0, column=depth_path, value=depth)
    if depth_path == "MD":
        df_output.insert(loc=1, column='TVDSS', value=depth_tvdss)
    # write file
    las.write(output_file, version=2)
    return df_output

def to_sgy(
    output_file, traces_x, traces_y, traces, il_const=1000, xl_const=2000,
):

    spec = segyio.spec()

    # issue warning if negative time/depth values
    if traces_y[0] < 0:
        print("WARNING: negative input sample values when writing segy file.")

    # get size of input
    ntr = len(traces_x)

    # specify structure
    spec = segyio.spec()
    spec.format = 1
    spec.sorting = 2
    spec.samples = traces_y
    spec.tracecount = ntr

    # add trace data
    with segyio.create(output_file, spec) as f:

        # loop over input traces
        for itr in np.arange(ntr):

            # trace header
            f.header[itr] = {
                segyio.su.offset: traces_x[itr],
                segyio.su.iline: il_const,
                segyio.su.xline: xl_const,
            }

            # trace data
            f.trace[itr] = traces[:, itr]

        f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)


def get_image_edges(xx, yy):

    # get image edges
    dy = yy[1] - yy[0]
    dx = xx[1] - xx[0]
    y_top = yy[0] - (dy / 2)
    y_base = yy[-1] + (dy / 2)
    x_left = xx[0] - (dx / 2)
    x_right = xx[-1] + (dx / 2)

    return [x_left, x_right, y_base, y_top]


def plot_image(
    traces_x,
    traces_y,
    traces,
    color_map_name="petrel_seismic_default",
    color_map_reverse=False,
    color_range_min=-0.3,
    color_range_max=0.3,
    int_method="bilinear",
    show_axis_ticks=False,
    upscaling_x=4,
    upscaling_y=8,
):

    # get colormap
    cmap = get_colormap(color_map_name, reverse=color_map_reverse)

    # parameters
    # int_method = "bilinear"
    sizes = np.shape(traces)

    # set figure size in pixels
    figure_width_pixels = upscaling_y * float(sizes[1])
    figure_height_pixels = upscaling_x * float(sizes[0])

    # create figure
    fig = plt.figure("SEISMIC TO PNG")
    fig.clf()

    # set figure size in inches
    figure_width_inches = figure_width_pixels / fig.dpi
    figure_height_inches = figure_height_pixels / fig.dpi
    fig.set_size_inches(figure_width_inches, figure_height_inches)

    # create axes
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    if show_axis_ticks:
        ax.set_axis_on()
        ax.tick_params(
            direction="in",
            length=6,
            width=1,
            colors="k",
            grid_color="k",
            grid_alpha=0.5,
        )
    ax.spines["top"].set_visible(False)
    fig.add_axes(ax)

    # get image edges
    extent = get_image_edges(traces_x, traces_y)

    # display image
    ax.imshow(
        traces,
        aspect="auto",
        cmap=cmap,
        vmin=color_range_min,
        vmax=color_range_max,
        interpolation=int_method,
        extent=extent,
    )

    # set axis limits
    plt.xlim(traces_x[0], traces_x[-1])
    plt.ylim(traces_y[-1], traces_y[0])


def get_image(
    traces_x,
    traces_y,
    traces,
    color_map_name="petrel_seismic_default",
    color_map_reverse=False,
    color_range_min=-0.3,
    color_range_max=0.3,
    int_method="bilinear",
    show_axis_ticks=False,
    upscaling_x=4,
    upscaling_y=8,
):

    # plot image
    plot_image(
        traces_x,
        traces_y,
        traces,
        color_map_name=color_map_name,
        color_map_reverse=color_map_reverse,
        color_range_min=color_range_min,
        color_range_max=color_range_max,
        int_method=int_method,
        show_axis_ticks=show_axis_ticks,
        upscaling_x=upscaling_x,
        upscaling_y=upscaling_y,
    )

    image_bytes = BytesIO()
    plt.savefig(image_bytes, format="png")
    image_bytes.seek(0)
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode()

    return image_base64


def to_png(
    output_file,
    traces_x,
    traces_y,
    traces,
    color_map_name="petrel_seismic_default",
    color_map_reverse=False,
    color_range_min=-0.3,
    color_range_max=0.3,
    int_method="bilinear",
    show_axis_ticks=False,
    upscaling_x=4,
    upscaling_y=8,
):

    # plot image
    plot_image(
        traces_x,
        traces_y,
        traces,
        color_map_name=color_map_name,
        color_map_reverse=color_map_reverse,
        color_range_min=color_range_min,
        color_range_max=color_range_max,
        int_method=int_method,
        show_axis_ticks=show_axis_ticks,
        upscaling_x=upscaling_x,
        upscaling_y=upscaling_y,
    )

    # export to file
    plt.savefig(output_file, format="png")