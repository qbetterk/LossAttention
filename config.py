import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        # self.vocab_path_train = './data/multi-woz-processed/vocab'
        # self.vocab_path_eval = None
        # self.data_path = './data/multi-woz-processed/'
        # self.data_file = 'data_for_damd.json'
        # self.dev_list = 'data/multi-woz/valListFile.json'
        # self.test_list = 'data/multi-woz/testListFile.json'
        # self.dbs = {
        #     'attraction': 'db/attraction_db_processed.json',
        #     'hospital': 'db/hospital_db_processed.json',
        #     'hotel': 'db/hotel_db_processed.json',
        #     'police': 'db/police_db_processed.json',
        #     'restaurant': 'db/restaurant_db_processed.json',
        #     'taxi': 'db/taxi_db_processed.json',
        #     'train': 'db/train_db_processed.json',
        # }
        # self.glove_path = './data/glove/glove.6B.50d.txt'
        # self.domain_file_path = 'data/multi-woz-processed/domain_files.json'
        # self.slot_value_set_path = 'db/value_set_processed.json'
        # self.multi_acts_path = 'data/multi-woz-processed/multi_act_mapping_train.json'
        # self.exp_path = 'to be generated'
        # self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        self.vocab_path_train = './data/multi-woz-processed/vocab'
        self.vocab_path_eval = None
        self.data_path = './data/multi-woz-processed/adapt_9/'

        self.domains = [
                        'attraction',
                        'train',
                        'taxi',
                        'restaurant',
                        'hospital',
                        'hotel',
                        'police'
                        ]
        self.source_domain = [
                                'attraction',
                                'train',
                                'taxi',
                                'hospital',
                                'hotel',
                                'police'
                                ]
        self.target_domain = 'restaurant'
        self.train_data_file = ['data_in_domain_' + domain + '.json' for domain in self.source_domain]
        self.adapt_data_file = 'adapt_data_in_domain_' + self.target_domain + '.json'
        self.test_data_file = 'test_data_in_domain_' + self.target_domain + '.json'

        self.dev_list = 'data/multi-woz/valListFile.json'
        self.test_list = 'data/multi-woz/testListFile.json'
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.domain_file_path = 'data/multi-woz-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.multi_acts_path = 'data/multi-woz-processed/multi_act_mapping_train.json'
        self.exp_path = '' #'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.spv_proportion = 100
        self.split = [9,1,0]

        # experiment settings
        self.degree_size = 5
        self.mode = 'unknown'
        self.cuda = True
        self.cuda_device = [4]
        self.exp_no = 'no_aug'
        self.seed = 333
        self.exp_domains = ['all']
        self.save_log = True
        self.report_interval = 5
        self.max_nl_length = 60
        self.max_span_length = 30
        self.truncated = False


        # model settings
        self.enc_layer_num = 1
        self.dec_layer_num = 1
        self.vocab_size = 3000
        self.embed_size = 50
        self.hidden_size = 100
        self.pointer_dim = 6 # fixed
        self.dropout = 0
        self.layer_norm = False
        self.skip_connect = False
        self.encoder_share = False
        self.attn_param_share = False
        self.copy_param_share = False
        self.enable_aspn = True
        self.use_pvaspn = False
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False

        # training settings
        self.lr = 0.005
        self.meta_lr = 0.005
        self.label_smoothing = .0
        self.lr_decay = 0.5
        self.batch_size = 32
        self.epoch_num = 500
        self.early_stop_count = 5
        self.weight_decay_count = 3
        self.teacher_force = 100
        self.multi_acts_training = False
        self.multi_act_sampling_num = 1
        self.valid_loss = 'score'

        # evaluation settings
        self.eval_load_path = '' #'to be generated'
        # self.eval_load_path ='experiments/maml2_all_no_aug_sd333_lr0.005_bs32_sp5_dc3'
        # self.eval_load_path ='experiments/filter3_all_no_aug_sd333_lr0.005_bs32_sp5_dc3'
        self.eval_per_domain = False
        self.use_true_pv_resp = True
        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_prev_dspn = True
        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = True
        self.use_true_db_pointer = False
        self.limit_bspn_vocab = False
        self.limit_aspn_vocab = False
        self.same_eval_as_cambridge = True
        self.same_eval_act_f1_as_hdsa = False
        self.aspn_decode_mode = 'greedy'  #beam, greedy, nucleur_sampling, topk_sampling
        self.beam_width = 5
        self.nbest = 5
        self.beam_diverse_param=0.2
        self.act_selection_scheme = 'high_test_act_f1'
        self.topk_num = 1
        self.nucleur_p = 0.
        self.record_mode = False



        #### mostly cared


        #### newly added
        self.token_weight = 0

        self.trans_enc_layer_num = 2
        self.trans_enc_head_num = 5
        self.trans_dec_layer_num = 3
        self.trans_dec_head_num = 4

        self.transformer_enc = 0
        self.transformer_dec = 0

        self.temp_var = False

        # # teacher model parameters
        self.t_layer_num = 1
        self.t_head_num = 2

        # # maximize teacher model's loss
        self.maxmin = False

        # # update model during validations
        self.val_update = True

        self.notes = ''
        self.add_to_fold_name = ''

        self.test_dir = ''

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self, log_dir = './log'):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if self.save_log:
            log_path = os.path.join(log_dir, 'log_{}_{}.txt'.format(self.mode, self.log_time))
            file_handler = logging.FileHandler(log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

