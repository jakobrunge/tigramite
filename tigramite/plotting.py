"""Tigramite plotting package."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

import numpy as np
import json, warnings, os, pathlib
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # python<=3.7
try:
    import matplotlib
    import networkx as nx
    with open(pathlib.Path(os.path.dirname(__file__)) / '../versions.py', 'r') as vfile:
        packages = json.loads(vfile.read())['all']
        packages = dict(map(lambda s: s.split('>='), packages))
        if metadata.version('matplotlib') < packages['matplotlib']:
            raise Exception('Version mismatch. Installed version of matplotlib', metadata.version('matplotlib'),
                          'Please install matplotlib>=', packages['matplotlib'])
        if metadata.version('networkx') < packages['networkx']:
            raise Exception('Version mismatch. Installed version of networkx', metadata.version('networkx'),
                          'Please install networkx>=', packages['networkx'])
except Exception as e:
    warnings.warn(str(e))

from matplotlib.colors import ListedColormap
import matplotlib.transforms as transforms
from matplotlib import pyplot, ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import sys
from operator import sub
import tigramite.data_processing as pp
from copy import deepcopy
import matplotlib.path as mpath
import matplotlib.patheffects as PathEffects

# TODO: Add proper docstrings to internal functions...


def _par_corr_trafo(cmi):
    """Transformation of CMI to partial correlation scale."""

    # Set negative values to small positive number
    # (zero would be interpreted as non-significant in some functions)
    if np.ndim(cmi) == 0:
        if cmi < 0.0:
            cmi = 1e-8
    else:
        cmi[cmi < 0.0] = 1e-8

    return np.sqrt(1.0 - np.exp(-2.0 * cmi))


def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""

    return -0.5 * np.log(1.0 - par_corr ** 2)


def _myround(x, base=5, round_mode="updown"):
    """Rounds x to a float with precision base."""

    if round_mode == "updown":
        return base * round(float(x) / base)
    elif round_mode == "down":
        return base * np.floor(float(x) / base)
    elif round_mode == "up":
        return base * np.ceil(float(x) / base)

    return base * round(float(x) / base)


def _make_nice_axes(ax, where=None, skip=2, color=None):
    """Makes nice axes."""

    if where is None:
        where = ["left", "bottom"]
    if color is None:
        color = {"left": "black", "right": "black", "bottom": "black", "top": "black"}

    if type(skip) == int:
        skip_x = skip_y = skip
    else:
        skip_x = skip[0]
        skip_y = skip[1]

    for loc, spine in ax.spines.items():
        if loc in where:
            spine.set_position(("outward", 5))  # outward by 10 points
            spine.set_color(color[loc])
            if loc == "left" or loc == "right":
                pyplot.setp(ax.get_yticklines(), color=color[loc])
                pyplot.setp(ax.get_yticklabels(), color=color[loc])
            if loc == "top" or loc == "bottom":
                pyplot.setp(ax.get_xticklines(), color=color[loc])
        elif loc in [
            item for item in ["left", "bottom", "right", "top"] if item not in where
        ]:
            spine.set_color("none")  # don't draw spine

        else:
            raise ValueError("unknown spine location: %s" % loc)

    # ax.xaxis.get_major_formatter().set_useOffset(False)

    # turn off ticks where there is no spine
    if "top" in where and "bottom" not in where:
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(ax.get_xticks()[::skip_x])
    elif "bottom" in where:
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xticks(ax.get_xticks()[::skip_x])
    else:
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_ticklabels([])
    if "right" in where and "left" not in where:
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(ax.get_yticks()[::skip_y])
    elif "left" in where:
        ax.yaxis.set_ticks_position("left")
        ax.set_yticks(ax.get_yticks()[::skip_y])
    else:
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_ticklabels([])

    ax.patch.set_alpha(0.0)


def _get_absmax(val_matrix):
    """Get value at absolute maximum in lag function array.
    For an (N, N, tau)-array this comutes the lag of the absolute maximum
    along the tau-axis and stores the (positive or negative) value in
    the (N,N)-array absmax."""

    absmax_indices = np.abs(val_matrix).argmax(axis=2)
    i, j = np.indices(val_matrix.shape[:2])

    return val_matrix[i, j, absmax_indices]


def _add_timeseries(
    fig,
    axes,
    i,
    time,
    dataseries,
    label,
    use_mask=False,
    mask=None,
    missing_flag=None,
    grey_masked_samples=False,
    data_linewidth=1.0,
    skip_ticks_data_x=1,
    skip_ticks_data_y=1,
    unit=None,
    last=False,
    time_label="",
    label_fontsize=10,
    color="black",
    grey_alpha=1.0,
):
    """Adds a time series plot to an axis.
    Plot of dataseries is added to axis. Allows for proper visualization of
    masked data.

    Parameters
    ----------
    fig : figure instance
        Figure instance.
    axes : axis instance
        Either gridded axis object or single axis instance.
    i : int
        Index of axis in gridded axis object.
    time : array
        Timelabel array.
    dataseries : array-like
        One-dimensional data series array of variable.
    missing_flag : number, optional (default: None)
        Flag for missing values in dataframe. Dismisses all time slices of
        samples where missing values occur in any variable and also flags
        samples for all lags up to 2*tau_max. This avoids biases, see section on
        masking in Supplement of [1]_.
    label : str
        Variable label.
    use_mask : bool, optional (default: False)
        Whether to use masked data.
    mask : array-like, optional (default: None)
        Data mask where True labels masked samples.
    grey_masked_samples : bool, optional (default: False)
        Whether to mark masked samples by grey fills ('fill') or grey data
        ('data').
    data_linewidth : float, optional (default: 1.)
        Linewidth.
    skip_ticks_data_x : int, optional (default: 1)
        Skip every other tickmark.
    skip_ticks_data_y : int, optional (default: 1)
        Skip every other tickmark.
    unit : str, optional (default: None)
        Units of variable.
    last : bool, optional (default: False)
        Specifiy whether this is the last panel where also the bottom axis is
        plotted.
    time_label : str, optional (default: '')
        Label of time axis.
    label_fontsize : int, optional (default: 10)
        Fontsize.
    color : str, optional (default: black)
        Line color.
    grey_alpha : float, optional (default: 1.)
        Opacity of line.
    """

    # axes[i].xaxis.get_major_formatter().set_useOffset(False)
    try:
        ax = axes[i]
    except:
        ax = axes

    if missing_flag is not None:
        dataseries_nomissing = np.ma.masked_where(
            dataseries == missing_flag, dataseries
        )
    else:
        dataseries_nomissing = np.ma.masked_where(
            np.zeros(dataseries.shape), dataseries
        )

    if use_mask:

        maskdata = np.ma.masked_where(mask, dataseries_nomissing)

        if grey_masked_samples == "fill":
            ax.fill_between(
                time,
                maskdata.min(),
                maskdata.max(),
                where=mask,
                color="grey",
                interpolate=True,
                linewidth=0.0,
                alpha=grey_alpha,
            )
        elif grey_masked_samples == "data":
            ax.plot(
                time,
                dataseries_nomissing,
                color="grey",
                marker=".",
                markersize=data_linewidth,
                linewidth=data_linewidth,
                clip_on=False,
                alpha=grey_alpha,
            )

        ax.plot(
            time,
            maskdata,
            color=color,
            linewidth=data_linewidth,
            marker=".",
            markersize=data_linewidth,
            clip_on=False,
        )
    else:
        ax.plot(
            time,
            dataseries_nomissing,
            color=color,
            linewidth=data_linewidth,
            clip_on=False,
        )

    if last:
        _make_nice_axes(
            ax, where=["left", "bottom"], skip=(skip_ticks_data_x, skip_ticks_data_y)
        )
        ax.set_xlabel(r"%s" % time_label, fontsize=label_fontsize)
    else:
        _make_nice_axes(ax, where=["left"], skip=(skip_ticks_data_x, skip_ticks_data_y))
    # ax.get_xaxis().get_major_formatter().set_useOffset(False)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.label_outer()

    ax.set_xlim(time[0], time[-1])

    trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    if unit:
        ax.set_ylabel(r"%s [%s]" % (label, unit), fontsize=label_fontsize)
    else:
        ax.set_ylabel(r"%s" % (label), fontsize=label_fontsize)

        # ax.text(.02, .5, r'%s [%s]' % (label, unit), fontsize=label_fontsize,
        #         horizontalalignment='left', verticalalignment='center',
        #         rotation=90, transform=trans)
    # else:
    #     ax.text(.02, .5, r'%s' % (label), fontsize=label_fontsize,
    #             horizontalalignment='left', verticalalignment='center',
    #             rotation=90, transform=trans)
    pyplot.tight_layout()


def plot_timeseries(
    dataframe=None,
    save_name=None,
    fig_axes=None,
    figsize=None,
    var_units=None,
    time_label="time",
    use_mask=False,
    grey_masked_samples=False,
    data_linewidth=1.0,
    skip_ticks_data_x=1,
    skip_ticks_data_y=2,
    label_fontsize=12,
):
    """Create and save figure of stacked panels with time series.

    Parameters
    ----------
    dataframe : data object, optional
        This is the Tigramite dataframe object. It has the attributes
        dataframe.values yielding a np array of shape (observations T,
        variables N) and optionally a mask of the same shape.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    fig_axes : subplots instance, optional (default: None)
        Figure and axes instance. If None they are created as
        fig, axes = pyplot.subplots(N,...)
    figsize : tuple of floats, optional (default: None)
        Figure size if new figure is created. If None, default pyplot figsize
        is used.
    var_units : list of str, optional (default: None)
        Units of variables.
    time_label : str, optional (default: '')
        Label of time axis.
    use_mask : bool, optional (default: False)
        Whether to use masked data.
    grey_masked_samples : bool, optional (default: False)
        Whether to mark masked samples by grey fills ('fill') or grey data
        ('data').
    data_linewidth : float, optional (default: 1.)
        Linewidth.
    skip_ticks_data_x : int, optional (default: 1)
        Skip every other tickmark.
    skip_ticks_data_y : int, optional (default: 2)
        Skip every other tickmark.
    label_fontsize : int, optional (default: 10)
        Fontsize of variable labels.
    """

    # Read in all attributes from dataframe
    data = dataframe.values
    mask = dataframe.mask
    var_names = dataframe.var_names
    missing_flag = dataframe.missing_flag
    datatime = dataframe.datatime

    T, N = data.shape

    if var_units is None:
        var_units = ["" for i in range(N)]

    if fig_axes is None:
        fig, axes = pyplot.subplots(N, sharex=True, figsize=figsize)
    else:
        fig, axes = fig_axes

    for i in range(N):
        if mask is None:
            mask_i = None
        else:
            mask_i = mask[:, i]
        _add_timeseries(
            fig=fig,
            axes=axes,
            i=i,
            time=datatime,
            dataseries=data[:, i],
            label=var_names[i],
            use_mask=use_mask,
            mask=mask_i,
            missing_flag=missing_flag,
            grey_masked_samples=grey_masked_samples,
            data_linewidth=data_linewidth,
            skip_ticks_data_x=skip_ticks_data_x,
            skip_ticks_data_y=skip_ticks_data_y,
            unit=var_units[i],
            last=(i == N - 1),
            time_label=time_label,
            label_fontsize=label_fontsize,
        )

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.95, hspace=0.3)
    pyplot.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)
    else:
        return fig, axes


def plot_lagfuncs(val_matrix, name=None, setup_args={}, add_lagfunc_args={}):
    """Wrapper helper function to plot lag functions.
    Sets up the matrix object and plots the lagfunction, see parameters in
    setup_matrix and add_lagfuncs.

    Parameters
    ----------
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    name : str, optional (default: None)
        File name. If None, figure is shown in window.
    setup_args : dict
        Arguments for setting up the lag function matrix, see doc of
        setup_matrix.
    add_lagfunc_args : dict
        Arguments for adding a lag function matrix, see doc of add_lagfuncs.

    Returns
    -------
    matrix : object
        Further lag functions can be overlaid using the
        matrix.add_lagfuncs(val_matrix) function.
    """

    N, N, tau_max_plusone = val_matrix.shape
    tau_max = tau_max_plusone - 1

    matrix = setup_matrix(N=N, tau_max=tau_max, **setup_args)
    matrix.add_lagfuncs(val_matrix=val_matrix, **add_lagfunc_args)

    if name is not None:
        matrix.savefig(name=name)

    return matrix


class setup_matrix:
    """Create matrix of lag function panels.
    Class to setup figure object. The function add_lagfuncs(...) allows to plot
    the val_matrix of shape (N, N, tau_max+1). Multiple lagfunctions can be
    overlaid for comparison.

    Parameters
    ----------
    N : int
        Number of variables
    tau_max : int
        Maximum time lag.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    figsize : tuple of floats, optional (default: None)
        Figure size if new figure is created. If None, default pyplot figsize
        is used.
    minimum : int, optional (default: -1)
        Lower y-axis limit.
    maximum : int, optional (default: 1)
        Upper y-axis limit.
    label_space_left : float, optional (default: 0.1)
        Fraction of horizontal figure space to allocate left of plot for labels.
    label_space_top : float, optional (default: 0.05)
        Fraction of vertical figure space to allocate top of plot for labels.
    legend_width : float, optional (default: 0.15)
        Fraction of horizontal figure space to allocate right of plot for
        legend.
    x_base : float, optional (default: 1.)
        x-tick intervals to show.
    y_base : float, optional (default: .4)
        y-tick intervals to show.
    plot_gridlines : bool, optional (default: False)
        Whether to show a grid.
    lag_units : str, optional (default: '')
    lag_array : array, optional (default: None)
        Optional specification of lags overwriting np.arange(0, tau_max+1)
    label_fontsize : int, optional (default: 10)
        Fontsize of variable labels.
    """

    def __init__(
        self,
        N,
        tau_max,
        var_names=None,
        figsize=None,
        minimum=-1,
        maximum=1,
        label_space_left=0.1,
        label_space_top=0.05,
        legend_width=0.15,
        legend_fontsize=10,
        x_base=1.0,
        y_base=0.5,
        plot_gridlines=False,
        lag_units="",
        lag_array=None,
        label_fontsize=10,
    ):

        self.tau_max = tau_max

        self.labels = []
        self.lag_units = lag_units
        # if lag_array is None:
        #     self.lag_array = np.arange(0, self.tau_max + 1)
        # else:
        self.lag_array = lag_array
        if x_base is None:
            self.x_base = 1
        else:
            self.x_base = x_base

        self.legend_width = legend_width
        self.legend_fontsize = legend_fontsize

        self.label_space_left = label_space_left
        self.label_space_top = label_space_top
        self.label_fontsize = label_fontsize

        self.fig = pyplot.figure(figsize=figsize)

        self.axes_dict = {}

        if var_names is None:
            var_names = range(N)

        plot_index = 1
        for i in range(N):
            for j in range(N):
                self.axes_dict[(i, j)] = self.fig.add_subplot(N, N, plot_index)
                # Plot process labels
                if j == 0:
                    trans = transforms.blended_transform_factory(
                        self.fig.transFigure, self.axes_dict[(i, j)].transAxes
                    )
                    self.axes_dict[(i, j)].text(
                        0.01,
                        0.5,
                        "%s" % str(var_names[i]),
                        fontsize=label_fontsize,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=trans,
                    )
                if i == 0:
                    trans = transforms.blended_transform_factory(
                        self.axes_dict[(i, j)].transAxes, self.fig.transFigure
                    )
                    self.axes_dict[(i, j)].text(
                        0.5,
                        0.99,
                        r"${\to}$ " + "%s" % str(var_names[j]),
                        fontsize=label_fontsize,
                        horizontalalignment="center",
                        verticalalignment="top",
                        transform=trans,
                    )

                # Make nice axis
                _make_nice_axes(
                    self.axes_dict[(i, j)], where=["left", "bottom"], skip=(1, 1)
                )
                if x_base is not None:
                    self.axes_dict[(i, j)].xaxis.set_major_locator(
                        ticker.FixedLocator(np.arange(0, self.tau_max + 1, x_base))
                    )
                    if x_base / 2.0 % 1 == 0:
                        self.axes_dict[(i, j)].xaxis.set_minor_locator(
                            ticker.FixedLocator(
                                np.arange(0, self.tau_max + 1, x_base / 2.0)
                            )
                        )
                if y_base is not None:
                    self.axes_dict[(i, j)].yaxis.set_major_locator(
                        ticker.FixedLocator(
                            np.arange(
                                _myround(minimum, y_base, "down"),
                                _myround(maximum, y_base, "up") + y_base,
                                y_base,
                            )
                        )
                    )
                    self.axes_dict[(i, j)].yaxis.set_minor_locator(
                        ticker.FixedLocator(
                            np.arange(
                                _myround(minimum, y_base, "down"),
                                _myround(maximum, y_base, "up") + y_base,
                                y_base / 2.0,
                            )
                        )
                    )

                    self.axes_dict[(i, j)].set_ylim(
                        _myround(minimum, y_base, "down"),
                        _myround(maximum, y_base, "up"),
                    )
                if j != 0:
                    self.axes_dict[(i, j)].get_yaxis().set_ticklabels([])
                self.axes_dict[(i, j)].set_xlim(0, self.tau_max)
                if plot_gridlines:
                    self.axes_dict[(i, j)].grid(
                        True,
                        which="major",
                        color="black",
                        linestyle="dotted",
                        dashes=(1, 1),
                        linewidth=0.05,
                        zorder=-5,
                    )

                plot_index += 1

    def add_lagfuncs(
        self,
        val_matrix,
        sig_thres=None,
        conf_matrix=None,
        color="black",
        label=None,
        two_sided_thres=True,
        marker=".",
        markersize=5,
        alpha=1.0,
    ):
        """Add lag function plot from val_matrix array.

        Parameters
        ----------
        val_matrix : array_like
            Matrix of shape (N, N, tau_max+1) containing test statistic values.
        sig_thres : array-like, optional (default: None)
            Matrix of significance thresholds. Must be of same shape as
            val_matrix.
        conf_matrix : array-like, optional (default: None)
            Matrix of shape (, N, tau_max+1, 2) containing confidence bounds.
        color : str, optional (default: 'black')
            Line color.
        label : str
            Test statistic label.
        two_sided_thres : bool, optional (default: True)
            Whether to draw sig_thres for pos. and neg. values.
        marker : matplotlib marker symbol, optional (default: '.')
            Marker.
        markersize : int, optional (default: 5)
            Marker size.
        alpha : float, optional (default: 1.)
            Opacity.
        """

        if label is not None:
            self.labels.append((label, color, marker, markersize, alpha))

        for ij in list(self.axes_dict):
            i = ij[0]
            j = ij[1]
            maskedres = np.copy(val_matrix[i, j, int(i == j) :])
            self.axes_dict[(i, j)].plot(
                range(int(i == j), self.tau_max + 1),
                maskedres,
                linestyle="",
                color=color,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                clip_on=False,
            )
            if conf_matrix is not None:
                maskedconfres = np.copy(conf_matrix[i, j, int(i == j) :])
                self.axes_dict[(i, j)].plot(
                    range(int(i == j), self.tau_max + 1),
                    maskedconfres[:, 0],
                    linestyle="",
                    color=color,
                    marker="_",
                    markersize=markersize - 2,
                    alpha=alpha,
                    clip_on=False,
                )
                self.axes_dict[(i, j)].plot(
                    range(int(i == j), self.tau_max + 1),
                    maskedconfres[:, 1],
                    linestyle="",
                    color=color,
                    marker="_",
                    markersize=markersize - 2,
                    alpha=alpha,
                    clip_on=False,
                )

            self.axes_dict[(i, j)].plot(
                range(int(i == j), self.tau_max + 1),
                np.zeros(self.tau_max + 1 - int(i == j)),
                color="black",
                linestyle="dotted",
                linewidth=0.1,
            )

            if sig_thres is not None:
                maskedsigres = sig_thres[i, j, int(i == j) :]

                self.axes_dict[(i, j)].plot(
                    range(int(i == j), self.tau_max + 1),
                    maskedsigres,
                    color=color,
                    linestyle="solid",
                    linewidth=0.1,
                    alpha=alpha,
                )
                if two_sided_thres:
                    self.axes_dict[(i, j)].plot(
                        range(int(i == j), self.tau_max + 1),
                        -sig_thres[i, j, int(i == j) :],
                        color=color,
                        linestyle="solid",
                        linewidth=0.1,
                        alpha=alpha,
                    )
        # pyplot.tight_layout()

    def savefig(self, name=None):
        """Save matrix figure.

        Parameters
        ----------
        name : str, optional (default: None)
            File name. If None, figure is shown in window.
        """

        # Trick to plot legend
        if len(self.labels) > 0:
            axlegend = self.fig.add_subplot(111, frameon=False)
            axlegend.spines["left"].set_color("none")
            axlegend.spines["right"].set_color("none")
            axlegend.spines["bottom"].set_color("none")
            axlegend.spines["top"].set_color("none")
            axlegend.set_xticks([])
            axlegend.set_yticks([])

            # self.labels.append((label, color, marker, markersize, alpha))
            for item in self.labels:
                label = item[0]
                color = item[1]
                marker = item[2]
                markersize = item[3]
                alpha = item[4]

                axlegend.plot(
                    [],
                    [],
                    linestyle="",
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    label=label,
                    alpha=alpha,
                )
            axlegend.legend(
                loc="upper left",
                ncol=1,
                bbox_to_anchor=(1.05, 0.0, 0.1, 1.0),
                borderaxespad=0,
                fontsize=self.legend_fontsize,
            ).draw_frame(False)

            self.fig.subplots_adjust(
                left=self.label_space_left,
                right=1.0 - self.legend_width,
                top=1.0 - self.label_space_top,
                hspace=0.35,
                wspace=0.35,
            )
            pyplot.figtext(
                0.5,
                0.01,
                r"lag $\tau$ [%s]" % self.lag_units,
                horizontalalignment="center",
                fontsize=self.label_fontsize,
            )
        else:
            self.fig.subplots_adjust(
                left=self.label_space_left,
                right=0.95,
                top=1.0 - self.label_space_top,
                hspace=0.35,
                wspace=0.35,
            )
            pyplot.figtext(
                0.55,
                0.01,
                r"lag $\tau$ [%s]" % self.lag_units,
                horizontalalignment="center",
                fontsize=self.label_fontsize,
            )

        if self.lag_array is not None:
            assert self.lag_array.shape == np.arange(self.tau_max + 1).shape
            for ij in list(self.axes_dict):
                i = ij[0]
                j = ij[1]
                self.axes_dict[(i, j)].set_xticklabels(self.lag_array[:: self.x_base])

        if name is not None:
            self.fig.savefig(name)
        else:
            pyplot.show()


def _draw_network_with_curved_edges(
    fig,
    ax,
    G,
    pos,
    node_rings,
    node_labels,
    node_label_size,
    node_alpha=1.0,
    standard_size=100,
    node_aspect=None,
    standard_cmap="OrRd",
    standard_color_links='black',
    standard_color_nodes='lightgrey',
    log_sizes=False,
    cmap_links="YlOrRd",
    # cmap_links_edges="YlOrRd",
    links_vmin=0.0,
    links_vmax=1.0,
    links_edges_vmin=0.0,
    links_edges_vmax=1.0,
    links_ticks=0.2,
    links_edges_ticks=0.2,
    link_label_fontsize=8,
    arrowstyle="->, head_width=0.4, head_length=1",
    arrowhead_size=3.0,
    curved_radius=0.2,
    label_fontsize=4,
    label_fraction=0.5,
    link_colorbar_label="link",
    # link_edge_colorbar_label='link_edge',
    inner_edge_curved=False,
    inner_edge_style="solid",
    network_lower_bound=0.2,
    show_colorbar=True,
    special_nodes=None,
):
    """Function to draw a network from networkx graph instance.
    Various attributes are used to specify the graph's properties.
    This function is just a beta-template for now that can be further
    customized.
    """

    from matplotlib.patches import FancyArrowPatch, Circle, Ellipse

    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])

    N = len(G)

    # This fixes a positioning bug in matplotlib.
    ax.scatter(0, 0, zorder=-10, alpha=0)

    def draw_edge(
        ax,
        u,
        v,
        d,
        seen,
        arrowstyle="->, head_width=0.4, head_length=1",
        outer_edge=True,
    ):

        # avoiding attribute error raised by changes in networkx
        if hasattr(G, "node"):
            # works with networkx 1.10
            n1 = G.node[u]["patch"]
            n2 = G.node[v]["patch"]
        else:
            # works with networkx 2.4
            n1 = G.nodes[u]["patch"]
            n2 = G.nodes[v]["patch"]

        # print("+++++++++++++++++++++++==cmap_links ", cmap_links)
        if outer_edge:
            rad = -1.0 * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["outer_edge_color"])
            else:
                if d["outer_edge_color"] is not None:
                    facecolor = d["outer_edge_color"]
                else:
                    facecolor = standard_color_links

            width = d["outer_edge_width"]
            alpha = d["outer_edge_alpha"]
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1.0
            arrowstyle = arrowstyle
            # link_edge = d['outer_edge_edge']
            linestyle = d.get("outer_edge_style")

            if d.get("outer_edge_attribute", None) == "spurious":
                facecolor = "grey"

            if d.get("outer_edge_type") in ["<-o", "<--", "<-x"]:
                n1, n2 = n2, n1

            if d.get("outer_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
                # "+->",
                # "<-+",
            ]:
                arrowstyle = "-"
                # linewidth = width*factor
            elif d.get("outer_edge_type") == "<->":
                arrowstyle = "<->, head_width=0.4, head_length=1"
                # linewidth = width*factor
            elif d.get("outer_edge_type") in ["o->", "-->", "<-o", "<--", "<-x", "x->", "+->", "<-+"]:
                arrowstyle = "->, head_width=0.4, head_length=1"

        else:
            rad = -1.0 * inner_edge_curved * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["inner_edge_color"])
            else:
                if d["inner_edge_color"] is not None:
                    facecolor = d["inner_edge_color"]
                else:
                    # print("HERE")
                    facecolor = standard_color_links

            width = d["inner_edge_width"]
            alpha = d["inner_edge_alpha"]

            if d.get("inner_edge_attribute", None) == "spurious":
                facecolor = "grey"
            # print(d.get("inner_edge_type"))
            if d.get("inner_edge_type") in ["<-o", "<--", "<-x", "<-+"]:
                n1, n2 = n2, n1

            if d.get("inner_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
            ]:
                arrowstyle = "-"
            elif d.get("inner_edge_type") == "<->":
                arrowstyle = "<->, head_width=0.4, head_length=1"
            elif d.get("inner_edge_type") in ["o->", "-->", "<-o", "<--", "<-x", "x->", "+->"]:
                arrowstyle = "->, head_width=0.4, head_length=1"

            linestyle = d.get("inner_edge_style")

        coor1 = n1.center
        coor2 = n2.center

        marker_size = width ** 2
        figuresize = fig.get_size_inches()

        # print("COLOR ", facecolor)
        e_p = FancyArrowPatch(
            coor1,
            coor2,
            arrowstyle=arrowstyle,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=width,
            lw=width / 2,
            alpha=alpha,
            linestyle=linestyle,
            color=facecolor,
            clip_on=False,
            patchA=n1,
            patchB=n2,
            shrinkA=0,
            shrinkB=0,
            zorder=-1,
        )

        ax.add_artist(e_p)
        path = e_p.get_path()
        vertices = path.vertices.copy()
        m, n = vertices.shape

        start = vertices[0]
        end = vertices[-1]

        # This must be added to avoid rescaling of the plot, when no 'o'
        # or 'x' is added to the graph.
        ax.scatter(*start, zorder=-10, alpha=0)

        if outer_edge:
            if d.get("outer_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)

        else:
            if d.get("inner_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                )
                ax.add_collection(circle_marker_end)

        if d["label"] is not None and outer_edge:
            # Attach labels of lags
            trans = None  # patch.get_transform()
            path = e_p.get_path()
            verts = path.to_polygons(trans)[0]
            if len(verts) > 2:
                label_vert = verts[1, :]
                l = d["label"]
                string = str(l)
                txt = ax.text(
                    label_vert[0],
                    label_vert[1],
                    string,
                    fontsize=link_label_fontsize,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=1,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="k")]
                )

        return rad

    # Collect all edge weights to get color scale
    all_links_weights = []
    all_links_edge_weights = []
    for (u, v, d) in G.edges(data=True):
        if u != v:
            if d["outer_edge"] and d["outer_edge_color"] is not None:
                all_links_weights.append(d["outer_edge_color"])
            if d["inner_edge"] and d["inner_edge_color"] is not None:
                all_links_weights.append(d["inner_edge_color"])

    if cmap_links is not None and len(all_links_weights) > 0:
        if links_vmin is None:
            links_vmin = np.array(all_links_weights).min()
        if links_vmax is None:
            links_vmax = np.array(all_links_weights).max()
        data_to_rgb_links = pyplot.cm.ScalarMappable(
            norm=None, cmap=pyplot.get_cmap(cmap_links)
        )
        data_to_rgb_links.set_array(np.array(all_links_weights))
        data_to_rgb_links.set_clim(vmin=links_vmin, vmax=links_vmax)
        # Create colorbars for links

        # setup colorbar axes.
        if show_colorbar:
            cax_e = pyplot.axes(
                [
                    0.55,
                    ax.get_subplotspec().get_position(ax.figure).bounds[1] + 0.02,
                    0.4,
                    0.025 + (len(all_links_edge_weights) == 0) * 0.035,
                ],
                frameon=False,
            )

            cb_e = pyplot.colorbar(
                data_to_rgb_links, cax=cax_e, orientation="horizontal"
            )
            # try:
            cb_e.set_ticks(
                np.arange(
                    _myround(links_vmin, links_ticks, "down"),
                    _myround(links_vmax, links_ticks, "up") + links_ticks,
                    links_ticks,
                )
            )
            # except:
            #     print('no ticks given')

            cb_e.outline.clear()
            cax_e.set_xlabel(
                link_colorbar_label, labelpad=1, fontsize=label_fontsize, zorder=-10
            )

    ##
    # Draw nodes
    ##
    node_sizes = np.zeros((len(node_rings), N))
    for ring in list(node_rings):  # iterate through to get all node sizes
        if node_rings[ring]["sizes"] is not None:
            node_sizes[ring] = node_rings[ring]["sizes"]

        else:
            node_sizes[ring] = standard_size
    max_sizes = node_sizes.max(axis=1)
    total_max_size = node_sizes.sum(axis=0).max()
    node_sizes /= total_max_size
    node_sizes *= standard_size

    def get_aspect(ax):
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # print(figW, figH)
        # Axis size on figure
        _, _, w, h = ax.get_position().bounds
        # Ratio of display units
        # print(w, h)
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        # print(data_ratio, disp_ratio)
        return disp_ratio / data_ratio

    if node_aspect is None:
        node_aspect = get_aspect(ax)

    # start drawing the outer ring first...
    for ring in list(node_rings)[::-1]:
        #        print ring
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string, 'vmin':float or None, 'vmax':float or None}}
        if node_rings[ring]["color_array"] is not None:
            color_data = node_rings[ring]["color_array"]
            if node_rings[ring]["vmin"] is not None:
                vmin = node_rings[ring]["vmin"]
            else:
                vmin = node_rings[ring]["color_array"].min()
            if node_rings[ring]["vmax"] is not None:
                vmax = node_rings[ring]["vmax"]
            else:
                vmax = node_rings[ring]["color_array"].max()
            if node_rings[ring]["cmap"] is not None:
                cmap = node_rings[ring]["cmap"]
            else:
                cmap = standard_cmap
            data_to_rgb = pyplot.cm.ScalarMappable(
                norm=None, cmap=pyplot.get_cmap(cmap)
            )
            data_to_rgb.set_array(color_data)
            data_to_rgb.set_clim(vmin=vmin, vmax=vmax)
            colors = [data_to_rgb.to_rgba(color_data[n]) for n in G]

            if node_rings[ring]["colorbar"]:
                # Create colorbars for nodes
                # cax_n = pyplot.axes([.8 + ring*0.11,
                # ax.get_subplotspec().get_position(ax.figure).bounds[1]+0.05, 0.025, 0.35], frameon=False) #
                # setup colorbar axes.
                # setup colorbar axes.
                cax_n = pyplot.axes(
                    [
                        0.05,
                        ax.get_subplotspec().get_position(ax.figure).bounds[1] + 0.02 + ring * 0.11,
                        0.4,
                        0.025 + (len(node_rings) == 1) * 0.035,
                    ],
                    frameon=False,
                )
                cb_n = pyplot.colorbar(data_to_rgb, cax=cax_n, orientation="horizontal")
                # try:
                cb_n.set_ticks(
                    np.arange(
                        _myround(vmin, node_rings[ring]["ticks"], "down"),
                        _myround(vmax, node_rings[ring]["ticks"], "up")
                        + node_rings[ring]["ticks"],
                        node_rings[ring]["ticks"],
                    )
                )
                # except:
                #     print ('no ticks given')
                cb_n.outline.clear()
                # cb_n.set_ticks()
                cax_n.set_xlabel(
                    node_rings[ring]["label"], labelpad=1, fontsize=label_fontsize
                )
        else:
            colors = None
            vmin = None
            vmax = None

        for n in G:
            if type(node_alpha) == dict:
                alpha = node_alpha[n]
            else:
                alpha = 1.0

            if special_nodes is not None:
                if n in special_nodes:
                    color_here = special_nodes[n]
                else:
                    color_here = 'grey'
            else:
                if colors is None:
                    color_here = standard_color_nodes
                else:
                    color_here = colors[n]

            c = Ellipse(
                pos[n],
                width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
                height=node_sizes[: ring + 1].sum(axis=0)[n],
                clip_on=False,
                facecolor=color_here,
                edgecolor=color_here,
                zorder=-ring - 1,
            )

            # else:
            #     if special_nodes is not None and n in special_nodes:
            #         color_here = special_nodes[n]
            #     else:
            #         color_here = colors[n]
            #     c = Ellipse(
            #         pos[n],
            #         width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
            #         height=node_sizes[: ring + 1].sum(axis=0)[n],
            #         clip_on=False,
            #         facecolor=colors[n],
            #         edgecolor=colors[n],
            #         zorder=-ring - 1,
            #     )

            ax.add_patch(c)

            # avoiding attribute error raised by changes in networkx
            if hasattr(G, "node"):
                # works with networkx 1.10
                G.node[n]["patch"] = c
            else:
                # works with networkx 2.4
                G.nodes[n]["patch"] = c

            if ring == 0:
                ax.text(
                    pos[n][0],
                    pos[n][1],
                    node_labels[n],
                    fontsize=node_label_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                    alpha=1.0,
                )

    # Draw edges
    seen = {}
    for (u, v, d) in G.edges(data=True):
        if d.get("no_links"):
            d["inner_edge_alpha"] = 1e-8
            d["outer_edge_alpha"] = 1e-8
        if u != v:
            if d["outer_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, arrowstyle, outer_edge=True)
            if d["inner_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, outer_edge=False)

    pyplot.subplots_adjust(bottom=network_lower_bound)


def plot_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    save_name=None,
    link_colorbar_label="MCI",
    node_colorbar_label="auto-MCI",
    link_width=None,
    link_attribute=None,
    node_pos=None,
    arrow_linewidth=10.0,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=0,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="OrRd",
    node_size=0.3,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    alpha=1.0,
    node_label_size=10,
    link_label_fontsize=10,
    lag_array=None,
    network_lower_bound=0.2,
    show_colorbar=True,
    inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
):
    """Creates a network plot.
    
    This is still in beta. The network is defined from links in graph. Nodes
    denote variables, straight links contemporaneous dependencies and curved
    arrows lagged dependencies. The node color denotes the maximal absolute
    auto-dependency and the link color the value at the lag with maximal
    absolute cross-dependency. The link label lists the lags with significant
    dependency in order of absolute magnitude. The network can also be
    plotted over a map drawn before on the same axis. Then the node positions
    can be supplied in appropriate axis coordinates via node_pos.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Must be of same shape as val_matrix. 
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    node_colorbar_label : str, optional (default: 'auto-MCI')
        Test statistic label for auto-dependencies.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    link_attribute : array-like, optional (default: None)
        String array of val_matrix.shape specifying link attributes.
    node_pos : dictionary, optional (default: None)
        Dictionary of node positions in axis coordinates of form
        node_pos = {'x':array of shape (N,), 'y':array of shape(N)}. These
        coordinates could have been transformed before for basemap plots.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    vmin_nodes : float, optional (default: 0)
        Node colorbar scale lower bound.
    vmax_nodes : float, optional (default: 1)
        Node colorbar scale upper bound.
    node_ticks : float, optional (default: 0.4)
        Node tick mark interval.
    cmap_nodes : str, optional (default: 'OrRd')
        Colormap for links.
    node_size : int, optional (default: 0.3)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    lag_array : array, optional (default: None)
        Optional specification of lags overwriting np.arange(0, tau_max+1)
    network_lower_bound : float, optional (default: 0.2)
        Fraction of vertical space below graph plot.
    show_colorbar : bool
        Whether to show colorbars for links and nodes.
    """

    if link_matrix is not None:
        raise ValueError("link_matrix is deprecated and replaced by graph array"
                         " which is now returned by all methods.")

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    graph = graph.squeeze()

    if graph.ndim == 4:
        raise ValueError("Time series graph of shape (N,N,tau_max+1,tau_max+1) cannot be represented by plot_graph,"
                         " use plot_time_series_graph instead.")

    if graph.ndim == 2:
        # If a non-time series (N,N)-graph is given, insert a dummy dimension
        graph = np.expand_dims(graph, axis = 2)

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
        cmap_nodes = None
    else:
        no_coloring = False

    (graph, val_matrix, link_width, link_attribute) = _check_matrices(
        graph, val_matrix, link_width, link_attribute)
    

    N, N, dummy = graph.shape
    tau_max = dummy - 1
    max_lag = tau_max + 1

    if np.count_nonzero(graph != "") == np.count_nonzero(
        np.diagonal(graph) != ""
    ):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(graph == "") == graph.size or diagonal:
        graph[0, 1, 0] = "---"
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # val_matrix[np.abs(val_matrix) < sig_thres] = 0.

    # Only draw link in one direction among contemp
    # Remove lower triangle
    link_matrix_upper = np.copy(graph)
    link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])

    # net = _get_absmax(link_matrix != "")
    net = np.any(link_matrix_upper != "", axis=2)
    G = nx.DiGraph(net)
    
    # This handels Graphs with no links.
    # nx.draw(G, alpha=0, zorder=-10)

    node_color = list(np.zeros(N))
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["no_links"] = no_links
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum
            argmax = np.abs(val_matrix[u, v][1:]).argmax() + 1
        else:
            argmax = 0

        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            # dic['inner_edge'] = ((np.abs(val_matrix[u, v][0]) >=
            #                       sig_thres[u, v][0]) or
            #                      (np.abs(val_matrix[v, u][0]) >=
            #                       sig_thres[v, u][0]))
            dic["inner_edge"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_type"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_alpha"] = alpha
            if no_coloring:
                dic["inner_edge_color"] = None
            else:
                dic["inner_edge_color"] = val_matrix[u, v, 0]
            # # value at argmax of average
            # if np.abs(val_matrix[u, v][0] - val_matrix[v, u][0]) > .0001:
            #     print("Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f" % (
            #           u, v, val_matrix[u, v][0], v, u, val_matrix[v, u][0]) +
            #           " due to conditions, finite sample effects or "
            #           "masking, here edge color = "
            #           "larger (absolute) value.")
            # dic['inner_edge_color'] = _get_absmax(
            #     np.array([[[val_matrix[u, v][0],
            #                    val_matrix[v, u][0]]]])).squeeze()

            if link_width is None:
                dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["inner_edge_width"] = (
                    link_width[u, v, 0] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                dic["inner_edge_attribute"] = None
            else:
                dic["inner_edge_attribute"] = link_attribute[u, v, 0]

            #     # fraction of nonzero values
            dic["inner_edge_style"] = "solid"
            # else:
            # dic['inner_edge_style'] = link_style[
            #         u, v, 0]

            all_strengths.append(dic["inner_edge_color"])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                # dic['outer_edge'] = np.any(
                #     np.abs(val_matrix[u, v][1:]) >= sig_thres[u, v][1:])
                dic["outer_edge"] = np.any(link_matrix_upper[u, v, 1:] != "")
            else:
                dic["outer_edge"] = False

            dic["outer_edge_type"] = link_matrix_upper[u, v, argmax]

            dic["outer_edge_alpha"] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = (
                    link_width[u, v, argmax] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                # fraction of nonzero values
                dic["outer_edge_attribute"] = None
            else:
                dic["outer_edge_attribute"] = link_attribute[u, v, argmax]

            # value at argmax of average
            if no_coloring:
                dic["outer_edge_color"] = None
            else:
                dic["outer_edge_color"] = val_matrix[u, v][argmax]
            all_strengths.append(dic["outer_edge_color"])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = np.abs(val_matrix[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            if lag_array is not None:
                dic["label"] = str([lag_array[l] for l in lags if l in sig_lags])[1:-1]
            else:
                dic["label"] = str([l for l in lags if l in sig_lags])[1:-1]
        else:
            # Node color is max of average autodependency
            if no_coloring:
                node_color[u] = None
            else:
                node_color[u] = val_matrix[u, v][argmax]
            dic["inner_edge_attribute"] = None
            dic["outer_edge_attribute"] = None

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    if special_nodes is not None:
        special_nodes_draw = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_draw[i] = special_nodes[node]
        special_nodes = special_nodes_draw
    

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    if node_pos is None:
        pos = nx.circular_layout(deepcopy(G))
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos["x"][i], node_pos["y"][i])

    if cmap_nodes is None:
        node_color = None

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": show_colorbar,
        }
    }

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes="lightgrey",
        standard_color_links="black",
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        # cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        # links_edges_ticks=.2, link_edge_colorbar_label='link_edge',
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize,
        link_colorbar_label=link_colorbar_label,
        network_lower_bound=network_lower_bound,
        show_colorbar=show_colorbar,
        # label_fraction=label_fraction,
        special_nodes=special_nodes,
    )

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax


def _reverse_patt(patt):
    """Inverts a link pattern"""

    if patt == "":
        return ""

    left_mark, middle_mark, right_mark = patt[0], patt[1], patt[2]
    if left_mark == "<":
        new_right_mark = ">"
    else:
        new_right_mark = left_mark
    if right_mark == ">":
        new_left_mark = "<"
    else:
        new_left_mark = right_mark

    return new_left_mark + middle_mark + new_right_mark

    # if patt in ['---', 'o--', '--o', 'o-o', '']:
    #     return patt[::-1]
    # elif patt == '<->':
    #     return '<->'
    # elif patt == 'o->':
    #     return '<-o'
    # elif patt == '<-o':
    #     return 'o->'
    # elif patt == '-->':
    #     return '<--'
    # elif patt == '<--':
    #     return '-->'


def _check_matrices(graph, val_matrix, link_width, link_attribute):

    if graph.dtype != "<U3":
        # Transform to new graph data type U3
        old_matrix = np.copy(graph)
        graph = np.zeros(old_matrix.shape, dtype="<U3")
        graph[:] = ""
        for i, j, tau in zip(*np.where(old_matrix)):
            if tau == 0:
                if old_matrix[j, i, 0] == 0:
                    graph[i, j, 0] = "-->"
                    graph[j, i, 0] = "<--"
                else:
                    graph[i, j, 0] = "o-o"
                    graph[j, i, 0] = "o-o"
            else:
                graph[i, j, tau] = "-->"
    elif graph.ndim == 4:
        pass
    else:
        # print(graph[:,:,0])
        # Assert that graph has valid and consistent lag-zero entries
        for i, j, tau in zip(*np.where(graph)):
            if tau == 0:
                if graph[i, j, 0] != _reverse_patt(graph[j, i, 0]):
                    raise ValueError(
                        "graph needs to have consistent lag-zero links (eg"
                        " graph[i,j,0]='-->' requires graph[j,i,0]='<--')"
                    )
                if (
                    val_matrix is not None
                    and val_matrix[i, j, 0] != val_matrix[j, i, 0]
                ):
                    raise ValueError("val_matrix needs to be symmetric for lag-zero")
                if (
                    link_width is not None
                    and link_width[i, j, 0] != link_width[j, i, 0]
                ):
                    raise ValueError("link_width needs to be symmetric for lag-zero")
                if (
                    link_attribute is not None
                    and link_attribute[i, j, 0] != link_attribute[j, i, 0]
                ):
                    raise ValueError(
                        "link_attribute needs to be symmetric for lag-zero"
                    )

            if graph[i, j, tau] not in [
                "---",
                "o--",
                "--o",
                "o-o",
                "o->",
                "<-o",
                "-->",
                "<--",
                "<->",
                "x-o",
                "o-x",
                "x--",
                "--x",
                "x->",
                "<-x",
                "x-x",
                "<-+",
                "+->",
            ]:
                raise ValueError("Invalid graph entry.")

    if val_matrix is None:
        # if graph.ndim == 4:
        #     val_matrix = (graph != "").astype("int")
        # else:
            val_matrix = (graph != "").astype("int")

    if link_width is not None and not np.all(link_width >= 0.0):
        raise ValueError("link_width must be non-negative")

    return graph, val_matrix, link_width, link_attribute


def plot_time_series_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    link_colorbar_label="MCI",
    save_name=None,
    link_width=None,
    link_attribute=None,
    arrow_linewidth=8,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    order=None,
    node_size=0.1,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=12,
    alpha=1.0,
    node_label_size=12,
    label_space_left=0.1,
    label_space_top=0.0,
    network_lower_bound=0.2,
    inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
    # aux_graph=None,
    standard_color_links='black',
    standard_color_nodes='lightgrey',
):
    """Creates a time series graph.
    This is still in beta. The time series graph's links are colored by
    val_matrix.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Either of shape (N, N, tau_max + 1) or as auxiliary graph of dims 
        (N, N, tau_max+1, tau_max+1) describing auxADMG. 
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    order : list, optional (default: None)
        order of variables from top to bottom.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    node_size : int, optional (default: 0.1)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    label_space_left : float, optional (default: 0.1)
        Fraction of horizontal figure space to allocate left of plot for labels.
    label_space_top : float, optional (default: 0.)
        Fraction of vertical figure space to allocate top of plot for labels.
    network_lower_bound : float, optional (default: 0.2)
        Fraction of vertical space below graph plot.
    inner_edge_style : string, optional (default: 'dashed')
        Style of inner_edge contemporaneous links.
    special_nodes : dict
        Dictionary of format {(i, -tau): 'blue', ...} to color special nodes.
    """

    if link_matrix is not None:
        raise ValueError("link_matrix is deprecated and replaced by graph array"
                         " which is now returned by all methods.")
        
    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
    else:
        no_coloring = False

    (graph, val_matrix, link_width, link_attribute) = _check_matrices(
        graph, val_matrix, link_width, link_attribute
    )

    if graph.ndim == 4:
        N, N, dummy, _ = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1
    else:
        N, N, dummy = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1

    if np.count_nonzero(graph == "") == graph.size:
        if graph.ndim == 4:
            graph[0, 1, 0, 0] = "---"
        else:
            graph[0, 1, 0] = "---"
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    if order is None:
        order = range(N)

    if set(order) != set(range(N)):
        raise ValueError("order must be a permutation of range(N)")

    def translate(row, lag):
        return row * max_lag + lag

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    tsg = np.zeros((N * max_lag, N * max_lag))
    tsg_val = np.zeros((N * max_lag, N * max_lag))
    tsg_width = np.zeros((N * max_lag, N * max_lag))
    tsg_style = np.zeros((N * max_lag, N * max_lag), dtype=graph.dtype)
    if link_attribute is not None:
        tsg_attr = np.zeros((N * max_lag, N * max_lag), dtype=link_attribute.dtype)

    # Only draw link in one direction
    # Remove lower triangle
    if graph.ndim == 4:
        for i, j, taui, tauj in np.column_stack(np.where(graph)):
            tau = taui - tauj
            if tau <= 0 and j <= i:
                continue
            # print(max_lag, (i, -taui), (j, -tauj), aux_graph[i, j, taui, tauj])
            # print(translate(i, max_lag - 1 - taui), translate(j, max_lag-1-tauj))
            tsg[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = 1.0
            tsg_val[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = 1. #val_matrix[i, j, tau]
            tsg_style[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = graph[i, j, taui, tauj]
            if link_width is not None:
                tsg_width[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = arrow_linewidth
            if link_attribute is not None:
                tsg_attr[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = 'spurious'
        # print(tsg_style)   
        # print(tsg)     

    else:
      link_matrix_tsg = np.copy(graph)
      link_matrix_tsg[:, :, 0] = np.triu(graph[:, :, 0])

      for i, j, tau in np.column_stack(np.where(link_matrix_tsg)):
        for t in range(max_lag):
            if (
                0 <= translate(i, t - tau)
                and translate(i, t - tau) % max_lag <= translate(j, t) % max_lag
            ):

                tsg[
                    translate(i, t - tau), translate(j, t)
                ] = 1.0  # val_matrix[i, j, tau]
                tsg_val[translate(i, t - tau), translate(j, t)] = val_matrix[i, j, tau]
                tsg_style[translate(i, t - tau), translate(j, t)] = graph[
                    i, j, tau
                ]
                if link_width is not None:
                    tsg_width[translate(i, t - tau), translate(j, t)] = (
                        link_width[i, j, tau] / link_width.max() * arrow_linewidth
                    )
                if link_attribute is not None:
                    tsg_attr[translate(i, t - tau), translate(j, t)] = link_attribute[
                        i, j, tau
                    ]
       

    G = nx.DiGraph(tsg)

    if special_nodes is not None:
        special_nodes_tsg = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_tsg[translate(i, max_lag-1 + tau)] = special_nodes[node]

        special_nodes = special_nodes_tsg

    # node_color = np.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["no_links"] = no_links
        if u != v:
            dic["inner_edge"] = False
            dic["outer_edge"] = True

            dic["outer_edge_type"] = tsg_style[u, v]

            dic["outer_edge_alpha"] = alpha

            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = dic["inner_edge_width"] = tsg_width[u, v]

            if link_attribute is None:
                dic["outer_edge_attribute"] = None
            else:
                dic["outer_edge_attribute"] = tsg_attr[u, v]

            # value at argmax of average
            if no_coloring:
                dic["outer_edge_color"] = None
            else:
                dic["outer_edge_color"] = tsg_val[u, v]

            all_strengths.append(dic["outer_edge_color"])
            dic["label"] = None
        # print(u, v, dic)

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1.0 - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        # for n in range(N):
        #     for tau in range(max_lag):
        #         i = n*N + tau
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - posarray.min(axis=0)[0])
                / (posarray.max(axis=0)[0] - posarray.min(axis=0)[0]),
                ((1.0 - i // max_lag) - posarray.min(axis=0)[1])
                / (posarray.max(axis=0)[1] - posarray.min(axis=0)[1]),
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {
        0: {"sizes": None, "color_array": None, "label": "", "colorbar": False,}
    }

    node_labels = ["" for i in range(N * max_lag)]

    if graph.ndim == 4:
        show_colorbar = False
    else:
        show_colorbar = True

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        node_rings=node_rings,
        node_labels=node_labels,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes=standard_color_nodes,
        standard_color_links=standard_color_links,
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        label_fraction=0.5,
        link_colorbar_label=link_colorbar_label,
        inner_edge_curved=True,
        network_lower_bound=network_lower_bound,
        inner_edge_style=inner_edge_style,
        special_nodes=special_nodes,
        show_colorbar=show_colorbar,
    )

    for i in range(N):
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(
            label_space_left,
            pos[order[i] * max_lag][1],
            f"{var_names[order[i]]}",
            fontsize=label_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t$",
                fontsize=int(label_fontsize * 0.8),
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )
        else:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t-%s$" % str(max_lag - tau - 1),
                fontsize=int(label_fontsize * 0.8),
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax


def plot_mediation_time_series_graph(
    path_node_array,
    tsg_path_val_matrix,
    var_names=None,
    fig_ax=None,
    figsize=None,
    link_colorbar_label="link coeff. (edge color)",
    node_colorbar_label="MCE (node color)",
    save_name=None,
    link_width=None,
    arrow_linewidth=8,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    order=None,
    vmin_nodes=-1.0,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="RdBu_r",
    node_size=0.1,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=12,
    alpha=1.0,
    node_label_size=12,
    label_space_left=0.1,
    label_space_top=0.0,
    network_lower_bound=0.2,
    standard_color_links='black',
    standard_color_nodes='lightgrey',
):
    """Creates a mediation time series graph plot.
    This is still in beta. The time series graph's links are colored by
    val_matrix.

    Parameters
    ----------
    tsg_path_val_matrix : array_like
        Matrix of shape (N*tau_max, N*tau_max) containing link weight values.
    path_node_array: array_like
        Array of shape (N,) containing node values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'link coeff. (edge color)')
        Link colorbar label.
    node_colorbar_label : str, optional (default: 'MCE (node color)')
        Node colorbar label.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    order : list, optional (default: None)
        order of variables from top to bottom.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    vmin_nodes : float, optional (default: 0)
        Node colorbar scale lower bound.
    vmax_nodes : float, optional (default: 1)
        Node colorbar scale upper bound.
    node_ticks : float, optional (default: 0.4)
        Node tick mark interval.
    cmap_nodes : str, optional (default: 'OrRd')
        Colormap for links.
    node_size : int, optional (default: 0.1)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    label_space_left : float, optional (default: 0.1)
        Fraction of horizontal figure space to allocate left of plot for labels.
    label_space_top : float, optional (default: 0.)
        Fraction of vertical figure space to allocate top of plot for labels.
    network_lower_bound : float, optional (default: 0.2)
        Fraction of vertical space below graph plot.
    """
    N = len(path_node_array)
    Nmaxlag = tsg_path_val_matrix.shape[0]
    max_lag = Nmaxlag // N

    if var_names is None:
        var_names = range(N)

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if link_width is not None and not np.all(link_width >= 0.0):
        raise ValueError("link_width must be non-negative")

    if order is None:
        order = range(N)

    if set(order) != set(range(N)):
        raise ValueError("order must be a permutation of range(N)")

    def translate(row, lag):
        return row * max_lag + lag

    if np.count_nonzero(tsg_path_val_matrix) == np.count_nonzero(
        np.diagonal(tsg_path_val_matrix)
    ):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(tsg_path_val_matrix) == tsg_path_val_matrix.size or diagonal:
        tsg_path_val_matrix[0, 1] = 1
        no_links = True
    else:
        no_links = False

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    tsg = tsg_path_val_matrix
    tsg_attr = np.zeros((N * max_lag, N * max_lag))

    G = nx.DiGraph(tsg)

    # node_color = np.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["no_links"] = no_links
        dic["outer_edge_attribute"] = None

        if u != v:

            if u % max_lag == v % max_lag:
                dic["inner_edge"] = True
                dic["outer_edge"] = False
            else:
                dic["inner_edge"] = False
                dic["outer_edge"] = True

            dic["inner_edge_alpha"] = alpha
            dic["inner_edge_color"] = _get_absmax(
                np.array([[[tsg[u, v], tsg[v, u]]]])
            ).squeeze()
            dic["inner_edge_width"] = arrow_linewidth
            all_strengths.append(dic["inner_edge_color"])

            dic["outer_edge_alpha"] = alpha

            dic["outer_edge_width"] = arrow_linewidth

            # value at argmax of average
            dic["outer_edge_color"] = tsg[u, v]
            all_strengths.append(dic["outer_edge_color"])
            dic["label"] = None

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1.0 - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        # for n in range(N):
        #     for tau in range(max_lag):
        #         i = n*N + tau
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - posarray.min(axis=0)[0])
                / (posarray.max(axis=0)[0] - posarray.min(axis=0)[0]),
                ((1.0 - i // max_lag) - posarray.min(axis=0)[1])
                / (posarray.max(axis=0)[1] - posarray.min(axis=0)[1]),
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_color = np.zeros(N * max_lag)
    for inet, n in enumerate(range(0, N * max_lag, max_lag)):
        node_color[n : n + max_lag] = path_node_array[inet]

    # node_rings = {0: {'sizes': None, 'color_array': color_array,
    #                   'label': '', 'colorbar': False,
    #                   }
    #               }

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": True,
        }
    }

    # ] for v in range(max_lag)]
    node_labels = ["" for i in range(N * max_lag)]

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=node_labels,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes=standard_color_nodes,
        standard_color_links=standard_color_links,
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        # cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        # links_edges_ticks=.2, link_edge_colorbar_label='link_edge',
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        label_fraction=0.5,
        link_colorbar_label=link_colorbar_label,
        inner_edge_curved=True,
        network_lower_bound=network_lower_bound
        # inner_edge_style=inner_edge_style
    )

    for i in range(N):
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(
            label_space_left,
            pos[order[i] * max_lag][1],
            "%s" % str(var_names[order[i]]),
            fontsize=label_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t$",
                fontsize=label_fontsize,
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )
        else:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t-%s$" % str(max_lag - tau - 1),
                fontsize=label_fontsize,
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )

    # fig.subplots_adjust(left=0.1, right=.98, bottom=.25, top=.9)
    # savestring = os.path.expanduser(save_name)
    if save_name is not None:
        pyplot.savefig(save_name)
    else:
        pyplot.show()


def plot_mediation_graph(
    path_val_matrix,
    path_node_array=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    save_name=None,
    link_colorbar_label="link coeff. (edge color)",
    node_colorbar_label="MCE (node color)",
    link_width=None,
    node_pos=None,
    arrow_linewidth=10.0,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=-1.0,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="RdBu_r",
    node_size=0.3,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    lag_array=None,
    alpha=1.0,
    node_label_size=10,
    link_label_fontsize=10,
    network_lower_bound=0.2,
    standard_color_links='black',
    standard_color_nodes='lightgrey',
):
    """Creates a network plot visualizing the pathways of a mediation analysis.
    This is still in beta. The network is defined from non-zero entries in
    ``path_val_matrix``.  Nodes denote variables, straight links contemporaneous
    dependencies and curved arrows lagged dependencies. The node color denotes
    the mediated causal effect (MCE) and the link color the value at the lag
    with maximal link coefficient. The link label lists the lags with
    significant dependency in order of absolute magnitude. The network can also
    be plotted over a map drawn before on the same axis. Then the node positions
    can be supplied in appropriate axis coordinates via node_pos.

    Parameters
    ----------
    path_val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing link weight values.
    path_node_array: array_like
        Array of shape (N,) containing node values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'link coeff. (edge color)')
        Link colorbar label.
    node_colorbar_label : str, optional (default: 'MCE (node color)')
        Node colorbar label.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    node_pos : dictionary, optional (default: None)
        Dictionary of node positions in axis coordinates of form
        node_pos = {'x':array of shape (N,), 'y':array of shape(N)}. These
        coordinates could have been transformed before for basemap plots.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    vmin_nodes : float, optional (default: 0)
        Node colorbar scale lower bound.
    vmax_nodes : float, optional (default: 1)
        Node colorbar scale upper bound.
    node_ticks : float, optional (default: 0.4)
        Node tick mark interval.
    cmap_nodes : str, optional (default: 'OrRd')
        Colormap for links.
    node_size : int, optional (default: 0.3)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    network_lower_bound : float, optional (default: 0.2)
        Fraction of vertical space below graph plot.
    lag_array : array, optional (default: None)
        Optional specification of lags overwriting np.arange(0, tau_max+1)
    """
    val_matrix = path_val_matrix

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if link_width is not None and not np.all(link_width >= 0.0):
        raise ValueError("link_width must be non-negative")

    N, N, dummy = val_matrix.shape
    tau_max = dummy - 1

    if np.count_nonzero(val_matrix) == np.count_nonzero(np.diagonal(val_matrix)):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(val_matrix) == val_matrix.size or diagonal:
        val_matrix[0, 1, 0] = 1
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # val_matrix[np.abs(val_matrix) < sig_thres] = 0.
    graph = val_matrix != 0.0
    net = _get_absmax(val_matrix)
    G = nx.DiGraph(net)

    node_color = np.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["outer_edge_attribute"] = None
        dic["no_links"] = no_links
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum
            argmax = np.abs(val_matrix[u, v][1:]).argmax() + 1
        else:
            argmax = 0
        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            # dic['inner_edge'] = ((np.abs(val_matrix[u, v][0]) >=
            #                       sig_thres[u, v][0]) or
            #                      (np.abs(val_matrix[v, u][0]) >=
            #                       sig_thres[v, u][0]))
            dic["inner_edge"] = graph[u, v, 0] or graph[v, u, 0]
            dic["inner_edge_alpha"] = alpha
            # value at argmax of average
            if np.abs(val_matrix[u, v][0] - val_matrix[v, u][0]) > 0.0001:
                print(
                    "Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f"
                    % (u, v, val_matrix[u, v][0], v, u, val_matrix[v, u][0])
                    + " due to conditions, finite sample effects or "
                    "masking, here edge color = "
                    "larger (absolute) value."
                )
            dic["inner_edge_color"] = _get_absmax(
                np.array([[[val_matrix[u, v][0], val_matrix[v, u][0]]]])
            ).squeeze()
            if link_width is None:
                dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["inner_edge_width"] = (
                    link_width[u, v, 0] / link_width.max() * arrow_linewidth
                )

            all_strengths.append(dic["inner_edge_color"])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                # dic['outer_edge'] = np.any(
                #     np.abs(val_matrix[u, v][1:]) >= sig_thres[u, v][1:])
                dic["outer_edge"] = np.any(graph[u, v, 1:])
            else:
                dic["outer_edge"] = False
            dic["outer_edge_alpha"] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = (
                    link_width[u, v, argmax] / link_width.max() * arrow_linewidth
                )

            # value at argmax of average
            dic["outer_edge_color"] = val_matrix[u, v][argmax]
            all_strengths.append(dic["outer_edge_color"])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = np.abs(val_matrix[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (np.where(graph[u, v, 1:])[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            if lag_array is not None:
                dic["label"] = str([lag_array[l] for l in lags if l in sig_lags])[1:-1]
            else:
                dic["label"] = str([l for l in lags if l in sig_lags])[1:-1]
        else:
            # Node color is max of average autodependency
            node_color[u] = val_matrix[u, v][argmax]

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    node_color = path_node_array
    # print node_color
    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    if node_pos is None:
        pos = nx.circular_layout(deepcopy(G))
    #            pos = nx.spring_layout(deepcopy(G))
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos["x"][i], node_pos["y"][i])

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": True,
        }
    }

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes=standard_color_nodes,
        standard_color_links=standard_color_links,
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        # cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        # links_edges_ticks=.2, link_edge_colorbar_label='link_edge',
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize,
        link_colorbar_label=link_colorbar_label,
        network_lower_bound=network_lower_bound,
        # label_fraction=label_fraction,
        # inner_edge_style=inner_edge_style
    )

    # fig.subplots_adjust(left=0.1, right=.9, bottom=.25, top=.95)
    # savestring = os.path.expanduser(save_name)
    if save_name is not None:
        pyplot.savefig(save_name)
    else:
        pyplot.show()


#
#  Functions to plot time series graphs from links including ancestors
#
def plot_tsg(links, X, Y, Z=None, anc_x=None, anc_y=None, anc_xy=None):
    """Plots TSG that is input in format (N*max_lag, N*max_lag).
    Compared to the tigramite plotting function here links
    X^i_{t-tau} --> X^j_t can be missing for different t'. Helpful to
    visualize the conditioned TSG.
    """

    def varlag2node(var, lag):
        """Translate from (var, lag) notation to node in TSG.
        lag must be <= 0.
        """
        return var * max_lag + lag

    def node2varlag(node):
        """Translate from node in TSG to (var, -tau) notation.
        Here tau is <= 0.
        """
        var = node // max_lag
        tau = node % (max_lag) - (max_lag - 1)
        return var, tau

    def _get_minmax_lag(links):
        """Helper function to retrieve tau_min and tau_max from links
        """

        N = len(links)

        # Get maximum time lag
        min_lag = np.inf
        max_lag = 0
        for j in range(N):
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                # func = link_props[2]
                if coeff != 0.:
                    min_lag = min(min_lag, abs(lag))
                    max_lag = max(max_lag, abs(lag))
        return min_lag, max_lag

    def _links_to_tsg(link_coeffs, max_lag=None):
        """Transform link_coeffs to time series graph.
        TSG is of shape (N*max_lag, N*max_lag).
        """
        N = len(link_coeffs)

        # Get maximum lag
        min_lag_links, max_lag_links = _get_minmax_lag(link_coeffs)

        # max_lag of TSG is max lag in links + 1 for the zero lag.
        if max_lag is None:
            max_lag = max_lag_links + 1

        tsg = np.zeros((N * max_lag, N * max_lag))

        for j in range(N):
            for link_props in link_coeffs[j]:
                i, lag = link_props[0]
                tau = abs(lag)
                coeff = link_props[1]
                # func = link_props[2]
                if coeff != 0.0:
                    for t in range(max_lag):
                        if (
                            0 <= varlag2node(i, t - tau)
                            and varlag2node(i, t - tau) % max_lag
                            <= varlag2node(j, t) % max_lag
                        ):
                            tsg[varlag2node(i, t - tau), varlag2node(j, t)] = 1.0

        return tsg

    color_list = ["lightgrey", "grey", "black", "red", "blue", "orange"]
    listcmap = ListedColormap(color_list)

    N = len(links)

    min_lag_links, max_lag_links = _get_minmax_lag(links)
    max_lag = max_lag_links

    for anc in X + Y:
        max_lag = max(max_lag, abs(anc[1]))
    for anc in Y:
        max_lag = max(max_lag, abs(anc[1]))
    if Z is not None:
        for anc in Z:
            max_lag = max(max_lag, abs(anc[1]))

    if anc_x is not None:
        for anc in anc_x:
            max_lag = max(max_lag, abs(anc[1]))
    if anc_y is not None:
        for anc in anc_y:
            max_lag = max(max_lag, abs(anc[1]))
    if anc_xy is not None:
        for anc in anc_xy:
            max_lag = max(max_lag, abs(anc[1]))

    max_lag = max_lag + 1

    tsg = _links_to_tsg(links, max_lag=max_lag)

    G = nx.DiGraph(tsg)

    figsize = (3, 3)
    link_colorbar_label = "MCI"
    arrow_linewidth = 8.0
    vmin_edges = -1
    vmax_edges = 1.0
    edge_ticks = 0.4
    cmap_edges = "RdBu_r"
    order = None
    node_size = .1
    arrowhead_size = 20
    curved_radius = 0.2
    label_fontsize = 10
    alpha = 1.0
    node_label_size = 10
    label_space_left = 0.1
    label_space_top = 0.0
    network_lower_bound = 0.2
    inner_edge_style = "dashed"

    node_color = np.ones(N * max_lag)  # , dtype = 'object')
    node_color[:] = 0

    if anc_x is not None:
        for n in [varlag2node(itau[0], max_lag - 1 + itau[1]) for itau in anc_x]:
            node_color[n] = 3
    if anc_y is not None:
        for n in [varlag2node(itau[0], max_lag - 1 + itau[1]) for itau in anc_y]:
            node_color[n] = 4
    if anc_xy is not None:
        for n in [varlag2node(itau[0], max_lag - 1 + itau[1]) for itau in anc_xy]:
            node_color[n] = 5

    for x in X:
        node_color[varlag2node(x[0], max_lag - 1 + x[1])] = 2
    for y in Y:
        node_color[varlag2node(y[0], max_lag - 1 + y[1])] = 2
    if Z is not None:
        for z in Z:
            node_color[varlag2node(z[0], max_lag - 1 + z[1])] = 1

    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_subplot(111, frame_on=False)
    var_names = range(N)
    order = range(N)

    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        if u != v:
            if tsg[u, v] and tsg[v, u]:
                dic["inner_edge"] = True
                dic["outer_edge"] = False
            else:
                dic["inner_edge"] = False
                dic["outer_edge"] = True

            dic["inner_edge_alpha"] = alpha
            dic["inner_edge_color"] = tsg[u, v]

            dic["inner_edge_width"] = arrow_linewidth
            dic["inner_edge_attribute"] = dic["outer_edge_attribute"] = None

            all_strengths.append(dic["inner_edge_color"])
            dic["outer_edge_alpha"] = alpha
            dic["outer_edge_width"] = dic["inner_edge_width"] = arrow_linewidth

            # value at argmax of average
            dic["outer_edge_color"] = tsg[u, v]

            all_strengths.append(dic["outer_edge_color"])
            dic["label"] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1.0 - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - posarray.min(axis=0)[0])
                / (posarray.max(axis=0)[0] - posarray.min(axis=0)[0]),
                ((1.0 - i // max_lag) - posarray.min(axis=0)[1])
                / (posarray.max(axis=0)[1] - posarray.min(axis=0)[1]),
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "label": "",
            "colorbar": False,
            "cmap": listcmap,
            "vmin": 0,
            "vmax": len(color_list),
        }
    }

    node_labels = ["" for i in range(N * max_lag)]

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        node_rings=node_rings,
        node_labels=node_labels,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=None,
        standard_cmap="OrRd",
        standard_color_links='black',
        standard_color_nodes='lightgrey',
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        label_fraction=0.5,
        link_colorbar_label=link_colorbar_label,
        inner_edge_curved=True,
        network_lower_bound=network_lower_bound,
        inner_edge_style=inner_edge_style,
    )

    for i in range(N):
        trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(
            label_space_left,
            pos[order[i] * max_lag][1],
            "%s" % str(var_names[order[i]]),
            fontsize=label_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t$",
                fontsize=int(label_fontsize * 0.7),
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )
        else:
            ax.text(
                pos[tau][0],
                1.0 - label_space_top,
                r"$t-%s$" % str(max_lag - tau - 1),
                fontsize=int(label_fontsize * 0.7),
                horizontalalignment="center",
                verticalalignment="top",
                transform=trans,
            )

    return fig, ax


if __name__ == "__main__":

    val_matrix = np.zeros((4, 4, 3))

    # Complete test case
    graph = np.zeros((3,3,2), dtype='<U3')

    graph[0, 1, 0] = "<-+"
    graph[1, 0, 0] = "+->"

    # graph[1, 2, 0] = "x->"
    # graph[2, 1, 0] = "<-x"

    # graph[0, 2, 0] = "x->"
    # graph[2, 0, 0] = "<-x"
    nolinks = np.zeros(graph.shape)
    # nolinks[range(4), range(4), 1] = 1

    # plot_time_series_graph(graph=nolinks)
    plot_graph(graph=graph, 
        save_name="/home/rung_ja/Downloads/tsg_test.pdf")

    # pyplot.show()

