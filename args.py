import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs
import time

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian Convolutional Encoder-Decoder Networks with SVGD')
        self.add_argument('--pre-trained', action='store_true', default=False, help='test using the pre-trained model or not.')
        self.add_argument('--net', type=str, default='RRDB', help='Net arch used. CBAM or RRDB')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=30, help='(20-30) number of model instances in SVGD')
        self.add_argument('--features', type=int, default=48, help='number of basic features in basic conv layers')

        # data
        self.add_argument('--data-dir', type=str, default="./datasets/", help='data directory')
        self.add_argument('--nmonth', type=int, default=225, help="number of months considered. For 2002.02-2020.12, nmonth=225")
        self.add_argument('--ntrain', type=int, default=144, help="number of training data")
        self.add_argument('--nt', type=int, default=3, help="number of lagged months")
        self.add_argument('--act-fun', type=str, default='PReLU', help="activation function, ReLU, SiLU, PReLU, mish")
        
        # training
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0025, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=0.01, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=12, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=20, help='batch size for testing')
        self.add_argument('--seed', type=int, default=None, help='manual seed used in Tensor')

        # logging
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded in post mode')

    def parse(self):
        args = self.parse_args()
        date = '/July_2'
        args.run_dir = args.exp_dir + date\
            + '/nsamples{}_ntrain{}_nt{}_nf{}_batch{}_lr{}_noiselr{}_epochs{}_{}_{}'.format(
                args.n_samples, args.ntrain, args.nt, args.features, args.batch_size, args.lr,
                args.lr_noise, args.epochs,args.net, args.act_fun)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])
        
        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))

        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
