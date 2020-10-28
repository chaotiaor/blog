import importlib
from lib.trainer.base_model import Base


def find_model_using_name(model_name):
    """Import the module "lib/trainer/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "lib.trainer." + model_name + "_model"
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, Base):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel"
              " with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_model(net, opt):
    """Create a model given the option.
    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        from models import create_model
        model = create_model(net, opt)
    """
    model = find_model_using_name(opt.trainer_name)
    instance = model(net, opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


if __name__ == '__main__':
    """
    """

