from pyro.nn import PyroModule


class Model(PyroModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_plates(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
