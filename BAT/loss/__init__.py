from ospark.nn.loss_function import LossFunction
from typing import Type
import importlib
import os


LOSS_REGISTRY_CLASS_NAME = set()
LOSS_REGISTRY = {}


class RegisterMeta(type):
    
    def __new__(mcls, name: str, base: tuple, attr: dict) -> Type[LossFunction]:
        cls = super().__new__(mcls, name, base, attr)
        if name not in LOSS_REGISTRY_CLASS_NAME:
            LOSS_REGISTRY_CLASS_NAME.add(name)
        else: 
            raise KeyError(f"The class {name} has been registered.")
        registered_name = attr["register_name"]
        if registered_name is not None and registered_name not in LOSS_REGISTRY:
            LOSS_REGISTRY[registered_name] = cls
        
        elif registered_name in register_type:
            raise KeyError(f"The register name {registered_name} is duplicated.")
                                                                                
        else:
            raise KeyError("register_name is not named")
        return cls

# Automatically import python files under BAT/loss.
for filename in os.listdir(os.path.dirname(__file__)):
    if "__" not in filename and ".py" in filename:
        module_name = filename.replace(".py", "")
        importlib.import_module("BAT.loss." + module_name)