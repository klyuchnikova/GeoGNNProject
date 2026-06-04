import torch
import argparse
import sys
import os

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000

        # data management
        self.dataset_file = self.resolve_data_path(args.dataset)
        self.friend_file = self.resolve_data_path(args.friendship)
        self.max_users = args.max_users  # 0 = use all available users
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.min_checkins = args.min_checkins

        # evaluation        
        self.validate_epoch = args.validate_epoch  # 每5轮验证一次
        self.report_user = args.report_user  # -1

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file  # 时间POI graph
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 空间POI graph
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    @staticmethod
    def resolve_data_path(path):
        if os.path.isabs(path) or os.sep in path or '/' in path:
            return path
        return './data/{}'.format(path)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')  # -1
        parser.add_argument('--hidden-dim', default=32, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')  # 100
        parser.add_argument('--rnn', default='gru', type=str, help='the recurrent implementation to use: [rnn|gru|lstm]')

        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        parser.add_argument('--max-users', default=0, type=int,
                            help='limit loaded users for a quick smoke test; 0 uses all users')
        parser.add_argument('--sequence-length', default=20, type=int,
                            help='fixed trajectory sequence length')
        parser.add_argument('--min-checkins', default=101, type=int,
                            help='minimum check-ins per user; use 5 * sequence_length + 1 for the original split')
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
                            help='training/evaluation log file prefix')
        parser.add_argument('--trans_loc_file', default='./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='POI transition graph pickle file')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='user graph pickle file')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='spatial POI graph pickle file')
        parser.add_argument('--trans_interact_file', default='./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='user-POI interaction graph pickle file')
        parser.add_argument('--use_weight', action='store_true', help='use W in AXW for graph convolution')
        parser.add_argument('--use_graph_user', action='store_true', help='use user graph')
        parser.add_argument('--use_spatial_graph', action='store_true', help='use spatial POI graph')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=200, type=int,  # 200
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')  # 1024
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def __str__(self):
        return (
                   'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)
