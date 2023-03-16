"""Handles the TerminalOut Class"""
# License: GNU General Public License v3.0
import ipywidgets as widgets
from ipywidgets import Widget


class TerminalOutput(Widget):
    """Shows the algorithm output"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_put = widgets.Output(
            layout={
                'border': '1px solid black',
                'min_width': '300px',
                'min_height': '300px',
                'max_height': '600px',
                'width': 'auto',
                'height': 'auto',
                'overflow': 'scroll'

            }
        )
        self.title = widgets.HTML(
            value="<H3>Terminal</H3>",
        )

    def show(self):
        """Shows the algorithm output"""
        return widgets.VBox([self.title, self.out_put])

    def get_output(self):
        """Gets the content of the output widget"""
        return self.out_put
