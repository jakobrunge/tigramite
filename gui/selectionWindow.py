"""Handles the SelectionWindow Class"""
# License: GNU General Public License v3.0
import ipywidgets as widgets
from ipywidgets import Widget
from gui import DropdownSelectionWidget, DataUploadWidgets


class SelectionWindow(Widget):
    """Handles the Selection functionality"""
    def __init__(self, data_path, tests, methods, test_parameters, method_parameters, plots, plot_parameters, **kwargs):
        super().__init__(**kwargs)
        self.data_widget = DataUploadWidgets(data_path, "Data:")
        self.test_widget = DropdownSelectionWidget(tests, tests[0], "Conditional Independence tests:",
                                                   test_parameters)
        self.methods_widget = DropdownSelectionWidget(methods, methods[0], "Methods:",
                                                      method_parameters)
        self.accordion = widgets.Accordion(
            children=[
                self.data_widget,
                self.test_widget.widget,
                self.methods_widget.widget,

            ])
        self.accordion.set_title(0, 'Data')
        self.accordion.set_title(1, 'Conditional Independence test')
        self.accordion.set_title(2, 'Method')

    def get_current_values(self):
        """Gets the current values from its attributes"""
        return self.data_widget.value, self.test_widget.get_value(), self.methods_widget.get_value(), \
                self.test_widget.get_parameter_values(), self.methods_widget.get_parameter_values()
