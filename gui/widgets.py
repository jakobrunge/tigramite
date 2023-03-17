"""This file contains all the widget functionality"""
# License: GNU General Public License v3.0
import os
import ipywidgets as widgets
from ipywidgets import Widget
from IPython.display import display


def DataUploadWidgets(path_of_data, name):
    """Widget used for data upload"""

    list_dir = []
    for entry in os.scandir(path_of_data):
        if ".npy" in entry.name:
            list_dir.append(entry.name)
    list_dir.append("none")
    return widgets.Dropdown(
        options=list_dir,
        value='none',
        description=name,
        disabled=False
    )

    #upload = widgets.FileUpload(accept='.npy', multiple=False)
    #print(upload)
    #return upload


class ResultAccordion(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def show(self):
        return

    def get_values(self):
        return


class AlphaLevelWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha_slider = widgets.FloatSlider(
            value=0.01,
            min=0.,
            max=1.,
            step=0.01,
            description='alpha_value:',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.01f',
        )

    def show(self):
        return self.alpha_slider

    def get_alpha_value(self):
        return self.alpha_slider.value


class ParameterSelectionWidget(Widget):
    """Parameter Selection Template; should have an accordion widget filled with a Drop-down widget displaying values"""

    def __init__(self, parameter_dict, **kwargs):
        super().__init__(**kwargs)
        self.current_parameter_widgets = []
        self.parameters = None
        self.add_parameter_button = widgets.Button(
            description="+"
        )
        self.parameter_dropdown = widgets.Dropdown(options=[])
        self.parameters_widget = widgets.VBox(children=[])
        self.parameter_accordion = widgets.Accordion(children=[])
        self.current_parameter = None
        self.used_parameters = []
        #self.parameter_accordion.set_title(0, "Parameter Selection")
        self.setup(parameter_dict)

    def setup(self, params):
        """Sets the different parameters up; call when update"""
        self.parameters = params
        values = []
        for x in self.parameters:
            values.append(x)
        self.parameter_dropdown.options = values
        self.parameter_dropdown.observe(self.on_change, names="value")
        self.parameters_widget.children = [widgets.HBox([self.parameter_dropdown, self.add_parameter_button])]
        self.parameter_accordion.children = [self.parameters_widget]
        self.current_parameter = values[0]
        self.add_parameter_button.on_click(self.add_parameter_to_parameter_widget)
        return self

    def add_parameter_to_parameter_widget(self, b):
        """Adds Paramter to parameter widget when added in the dropdown"""
        parameter = Parameter(self.parameters[self.current_parameter])
        self.current_parameter_widgets.append(parameter.widget)
        self.used_parameters.append(parameter)
        self.parameters_widget.children = tuple(list(self.parameters_widget.children) + [parameter.widget])

    def on_change(self, change):
        """Called when parameter dropdown value is changed"""
        if change['type'] == 'change' and change['name'] == 'value':
            self.current_parameter = change['new']

    def get_currently_selected_parameters(self):
        """returns the currently selected parameters the user want to add to the calculation, turns them into a
        dictionary"""
        dict_values = {}
        for para in self.used_parameters:
            name, value = para.get_current_value()
            dict_values[name] = value
        return dict_values

    def update(self, params):
        """Updates the dropdown menu with the new parameters"""
        self.used_parameters = []
        self.parameters_widget.children = [widgets.HBox([self.parameter_dropdown, self.add_parameter_button])]
        self.setup(params)


class Parameter(Widget):
    """Handles parameter functionality and data type"""
    def __init__(self, parameter, **kwargs):
        super().__init__(**kwargs)
        self.name = '%s' % parameter["name"]
        self.para_dict = parameter
        self.widget = None
        try:
            if parameter["dtype"] == "int":
                self.widget = widgets.IntSlider(description=self.name)
            elif parameter["dtype"] == "float":
                self.widget = widgets.FloatSlider(description=self.name)
            elif parameter["dtype"] == "str":
                self.widget = widgets.Text(description=self.name)
            elif parameter["dtype"] == "bool":
                self.widget = widgets.Checkbox(description=self.name)
            elif parameter["dtype"] == "dict":
                self.widget = widgets.Text(description=self.name)
            elif parameter["dtype"] == "selection":
                self.widget = widgets.Dropdown(description=self.name, options=parameter["selection"])
        except Exception as e:
            print(str(e))

    def get_current_value(self):
        """return current name and value from the parameter"""
        return self.para_dict["name"], self.widget.value


class DropdownSelectionWidget(Widget):
    """Handles functionality for Method and Parameter Selection; consists of a Dropdown Menu and a
    ParameterAccordion instance"""

    def __init__(self, options, value, description, parameter_dict, **kwargs):
        super().__init__(**kwargs)
        self.drop_down = widgets.Dropdown(
            options=options,
            value=value,
            description=description,
            disabled=False
        )
        self.parameter_dict = parameter_dict
        self.drop_down.observe(self.on_change, names="value")
        self.parameter_accordion = ParameterSelectionWidget(self.parameter_dict[self.drop_down.value])
        self.widget = widgets.VBox([self.drop_down, self.parameter_accordion.parameter_accordion])

    def get_value(self):
        """Gets the current dropdown value"""
        return self.drop_down.value

    def on_change(self, change):
        """Calls the update after dropdown item is changed"""
        if change['type'] == 'change' and change['name'] == 'value':
            self.parameter_accordion.update(self.parameter_dict[change['new']])

    def get_parameter_values(self):
        """Gets the parameters values"""
        return self.parameter_accordion.get_currently_selected_parameters()


def plots_widget(plots, value):
    """Dropdown widget for plot selection"""
    return widgets.Dropdown(
        options=plots,
        value=value,
        description="Plot:",
        disabled=False,
        continuous_update=False,
    )


def run_button_widget():
    """run button"""
    return widgets.Button(description="Run")


def plots_show_button():
    """show button"""
    return widgets.Button(description="Show")
