"""Handles the PlotOut Class"""
# License: GNU General Public License v3.0
import ipywidgets as widgets
from ipywidgets import Widget
from gui import plots_widget, AlphaLevelWidget, DropdownSelectionWidget
from gui import constants


class PlotOut(Widget):
    """Handles plotting functionality"""
    def __init__(self, plots, plot_parameters, **kwargs):
        super().__init__(**kwargs)
        self.out = widgets.Output(
            layout={
                'border': '1px solid black',
                'min_width': '300px',
                'min_height': '300px',
                'max_height': '600px',
                'width': 'auto',
                'height': 'auto',
                'overflow': 'scroll'

            })
        self.plot_selection_widget = DropdownSelectionWidget(plots, plots[0], "Plots: ", plot_parameters)
        self.title = widgets.HTML(
            value="<H3>Plots</H3>",
        )
        self.alpha_level = AlphaLevelWidget()

    def show(self):
        """Shows the output widget"""
        return widgets.VBox([self.title, self.out, self.plot_selection_widget.widget, self.alpha_level.show()])

    def get_current_values(self):
        """Returns current values"""
        return self.plot_selection_widget.get_value(), self.alpha_level.get_alpha_value(), \
               self.plot_selection_widget.get_parameter_values()

    def get_output(self):
        """Gets output from output widget"""
        return self.out
