import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import base64
from io import BytesIO
import segyio
import lasio


def resample_log(zz, val, zz_out):
    ff = interp1d(zz, val, kind="next", bounds_error=False, fill_value=np.nan)
    return ff(zz_out)

def to_las(depth_path, uwi,output_file, depth_in,vp_input,vp_ext,vp_output,depth_step,null_value=-999.25):

    # create output depth curve
    if depth_step is not False:
        depth = np.arange(depth_in[0], depth_in[-1] + depth_step, depth_step)
        depth = depth[depth <= depth_in[-1]]
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
    if depth_path == "MD":
        las.append_curve("DEPTH", depth, unit="m", descr="Measured Depth")
    elif depth_path == "TVDSS":
        las.append_curve("DEPTH", depth, unit="m", descr="True Vertical Depth SubSea")

    #las.add_curve("TVD", depth, unit="m", descr="True Vertical Depth KB")
    #las.add_curve("TVDMSL", depth, unit="m", descr="True Vertical Depth Mean Sea Level")

    # loop through data columns and add
    # las.add_curve('VP', resample_log(depth_in, vp, depth), unit='m/s', descr = 'P-velocity')
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

            # add to las
            las.append_curve(
                log_name[ii].upper(), val, unit=log_unit[ii], descr=log_descr[ii]
            )

    # write file
    las.write(output_file, version=2)


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