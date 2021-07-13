import os
import pathlib


class Config:
    def __init__(self):

        self.device = None
        self.dataset = "HiC_Rao_10kb"
        self.model = "GraphTransformer"
        self.num_chr = 23
        self.genome_len = 288091
        self.resolution = 10000
        self.max_norm = 10
        self.lstm_nontrain = False

        self.gpu = {
            "use": False,
            "id": 0
        }

        ##########################################
        ############ Model Parameters ############
        ##########################################

        self.net_params = {
            "L": 4,
            "n_heads": 8,
            "hidden_dim": 16,
            "out_dim": 16,
            "edge_feat": True,
            "residual": True,
            "readout": "mean",
            "in_feat_dropout": 0.0,
            "dropout": 0.0,
            "layer_norm": False,
            "batch_norm": True,
            "self_loop": False,
            "lap_pos_enc": False,
            "wl_pos_enc": False,
            "full_graph": False,
            "device": None
        }

        self.params = {
            "seed": 41,
            "epochs": 4,
            "batch_size": 8,
            "init_lr": 0.0007,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 15,
            "min_lr": 1e-6,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
            "max_time": 24
        }

        ##########################################
        ############ Input Directories ###########
        ##########################################

        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.start_end_file = 'starts.npy'

        ##########################################
        ############ Output Locations ############
        ##########################################

        self.model_dir = '../saved_models/'
        self.output_directory = '../outputs/'
        self.plot_dir = self.output_directory + 'data_plots/'
        self.processed_data_dir = self.output_directory + 'processed_data/'

        for file_path in [self.model_dir, self.output_directory, self.plot_dir, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)
