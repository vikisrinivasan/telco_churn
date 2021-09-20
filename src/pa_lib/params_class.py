from typing import *


class Parameters(object):
    def __init__(self, yaml_file: Optional[str]):
        from yaml import load
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        if yaml_file is None:
            self.attributes = []
        else:
            with open(yaml_file, 'r') as f:
                params = load(f, Loader=Loader)
                for var in params:
                    setattr(self, var, params[var])
                self.attributes = params.keys()

    def __len__(self) -> int:
        return len(self.attributes)

    def __repr__(self) -> str:
        return ('{0}({1})'.format(self.__class__.__name__,
                                  dict((attr, getattr(self, attr)) for attr in self.attributes)))

    def get_param(self, variable: str) -> Any:
        if (variable not in self.attributes):
            return ValueError(f'the parameter {variable} provided does not exist')
        else:
            return getattr(self, variable)
