import argparse
import os
from lib.util import util


class BaseOptions:
    def __init__(self):
        """Reset the class;
        indicates the class hasn't been initailized
        """
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str,
                            default='C:\\Users\\mwrd\\Desktop\\old\\eyelid_gland_data\\data')
        parser.add_argument('--trainer_name', type=str, default='portrait',
                            help='portrait|bise')
        parser.add_argument('--net_name', type=str, default='portraitnet',
                            help='portraitnet|bisenet')
        parser.add_argument('--context_path', type=str, default='resnet18',
                            help='resnet18|resnet101')
        parser.add_argument('--model_name', type=str, default='G')
        parser.add_argument('--size', type=tuple, default=(352, 256))
        parser.add_argument('--n_class', type=int, default=3,
                            help='')
        parser.add_argument('--width_mult', type=int, default=1.0,
                            help='')

        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=list, default=[],
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models are saved here')

        parser.add_argument('--serial_batches', type=bool, default=False,
                            help='if true, takes images in order to make batches, '
                                 'otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0,'
                                 ' the code will load models by iter_[load_iter]; '
                                 'otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # 模型参数
        parser.add_argument('--useUpsample', type=bool, default=False,
                            help='')
        parser.add_argument('--useDeconvGroup', type=bool, default=False,
                            help='')
        parser.add_argument('--addEdge', type=bool, default=True,
                            help='')
        parser.add_argument('--weightInit', type=bool, default=True,
                            help='')
        parser.add_argument('--video', type=bool, default=False,
                            help='')
        parser.add_argument('--minChannel', type=int, default=16,
                            help='')
        parser.add_argument('--channelRatio', type=float, default=1.0,
                            help='')

        # loss 类别
        parser.add_argument('--loss_type', type=str, default='sigmoid',
                            help='softmax, sigmoid, focal, dice')
        # 数据划分
        parser.add_argument('--balance_dict', type=dict, default={'0': 1, '1': 3, '2': 10},
                            help='')

        parser.add_argument('--angle_range', type=tuple, default=(-45, 45),
                            help='')
        parser.add_argument('--scale_range', type=tuple, default=(0.5, 1.5),
                            help='')
        parser.add_argument('--temperature', type=int, default=1,
                            help='')
        parser.add_argument('--metric', type=int, default=0,
                            help='')
        parser.add_argument('--alpha', type=list, default=[1, 1, 1],
                            help='')
        parser.add_argument('--gamma', type=float, default=1.0,
                            help='')
        parser.add_argument('--size_average', type=bool, default=True,
                            help='')
        parser.add_argument('--shrink', type=float, default=0.2,
                            help='')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir, False)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints_9_16 directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt


