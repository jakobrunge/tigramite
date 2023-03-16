"""Module to be imported in the Jupyter Notebook"""
# License: GNU General Public License v3.0
import ipywidgets as widgets
import gui.selectionWindow as sw
from gui.plotWindow import PlotOut
from gui.widgets import run_button_widget, plots_show_button
from gui.methods import calculate_results, make_plot
from gui.terminalWindow import TerminalOutput
import gui.constants as constants


class ProjectWindow:
    """Project Window should contain: SelectionWindow, PlotWindow, TerminalWindow"""

    def __init__(self, data_path=constants.DATA_PATH, tests=constants.TESTS, methods=constants.METHODS,
                 test_parameter=constants.TEST_PARAMETER, method_parameters=constants.METHOD_PARAMETER,
                 plots=constants.PLOTS, plot_parameter=constants.PLOT_PARAMETER):
        self.plots_button = plots_show_button()
        self.SelectionWindow = sw.SelectionWindow(data_path, tests, methods, test_parameter, method_parameters, plots, plot_parameter)
        self.run_button = run_button_widget()
        self.results = None
        self.pcmci = None
        self.run_button.on_click(self.on_run_button_clicked)
        self.terminal_out = TerminalOutput()
        self.plot_out = PlotOut(plots, plot_parameter)
        self.plots_button.on_click(self.on_plot_button_clicked)

    def show(self):
        """Shows the ProjectWindow in the Jupyter Notebook"""
        return widgets.VBox(
            [
                self.SelectionWindow.accordion,
                widgets.HBox([self.run_button]),
                widgets.HBox([self.terminal_out.show(),
                              widgets.VBox(
                                  [self.plot_out.show(), self.plots_button]
                              )])
            ])

    def on_run_button_clicked(self, b):
        """Handles the functionality when the Run button is clicked"""
        data, test, method, test_params, method_params = self.SelectionWindow.get_current_values()
        self.pcmci, self.results = calculate_results(data, test, method, test_params, method_params,
                                                     self.terminal_out.get_output())

    def on_plot_button_clicked(self, b):
        """Handles the functionality when the Plot button is clicked"""
        plot_type, alpha_value, plot_parameters = self.plot_out.get_current_values()
        result = make_plot(plot_type, self.pcmci, self.results, self.plot_out.get_output(), alpha_value, plot_parameters)
