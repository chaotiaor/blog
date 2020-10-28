from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.set_defaults(batch_size=1)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False
        return parser










