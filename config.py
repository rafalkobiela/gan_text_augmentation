import os
import re
import time
import torch
from time import strftime, localtime


class Config():

    def __init__(self):
        self.if_test = False
        self.CUDA = True
        self.if_save = True
        self.data_shuffle = False  # False
        self.oracle_pretrain = True  # True
        self.gen_pretrain = False
        self.dis_pretrain = False
        self.clas_pretrain = False
        self.ora_pretrain = 1

        self.run_model = 'seqgan'  # seqgan, leakgan, maligan, jsdgan, relgan, sentigan
        self.k_label = 2  # num of labels, >=2
        self.gen_init = 'truncated_normal'  # normal, uniform, truncated_normal
        self.dis_init = 'uniform'  # normal, uniform, truncated_normal

        self.if_real_data = True  # if use real data
        self.dataset = 'image_coco'  # oracle, image_coco, emnlp_news, amazon_app_book, mr15
        self.model_type = 'vanilla'  # vanilla, RMC (custom)
        self.loss_type = 'rsgan'  # standard, JS, KL, hinge, tv, LS, rsgan (for RelGAN)
        self.vocab_size = 5000  # oracle: 5000, coco: 6613, emnlp: 5255, amazon_app_book: 6418, mr15: 6289
        self.max_seq_len = 20  # oracle: 20, coco: 37, emnlp: 51, amazon_app_book: 40
        self.ADV_train_epoch = 2000  # SeqGAN, LeakGAN-200, RelGAN-3000
        self.extend_vocab_size = 0  # plus test data, only used for Classifier

        self.temp_adpt = 'exp'  # no, lin, exp, log, sigmoid, quad, sqrt
        self.temperature = 1

        # ===Basic Train===
        self.samples_num = 10000  # 10000, mr15: 2000,
        self.MLE_train_epoch = 150  # SeqGAN-80, LeakGAN-8, RelGAN-150
        self.PRE_clas_epoch = 10
        self.inter_epoch = 15  # LeakGAN-10
        self.batch_size = 64  # 64
        self.start_letter = 1
        self.padding_idx = 0
        self.start_token = 'BOS'
        self.padding_token = 'EOS'
        self.gen_lr = 0.01  # 0.01
        self.gen_adv_lr = 1e-4  # RelGAN-1e-4
        self.dis_lr = 1e-4  # SeqGAN,LeakGAN-1e-2, RelGAN-1e-4
        self.clas_lr = 1e-3
        self.clip_norm = 5.0

        self.pre_log_step = 10
        self.adv_log_step = 20

        self.train_data = 'data/' + self.dataset + '.txt'
        self.test_data = 'data/testdata/' + self.dataset + '_test.txt'
        self.cat_train_data = 'data/' + self.dataset + '_cat{}.txt'
        self.cat_test_data = 'data/testdata/' + self.dataset + '_cat{}_test.txt'

        # ===Metrics===
        self.use_nll_oracle = True
        self.use_nll_gen = True
        self.use_nll_div = True
        self.use_bleu = True
        self.use_self_bleu = True
        self.use_clas_acc = True
        self.use_ppl = False

        # ===Generator===
        self.ADV_g_step = 1  # 1
        self.rollout_num = 16  # 4
        self.gen_embed_dim = 32  # 32
        self.gen_hidden_dim = 32  # 32
        self.goal_size = 16  # LeakGAN-16
        self.step_size = 4  # LeakGAN-4

        self.mem_slots = 1  # RelGAN-1
        self.num_heads = 2  # RelGAN-2
        self.head_size = 256  # RelGAN-256

        # ===Discriminator===
        self.d_step = 5  # SeqGAN-50, LeakGAN-5
        self.d_epoch = 3  # SeqGAN,LeakGAN-3
        self.ADV_d_step = 5  # SeqGAN,LeakGAN,RelGAN-5
        self.ADV_d_epoch = 3  # SeqGAN,LeakGAN-3

        self.dis_embed_dim = 64
        self.dis_hidden_dim = 64
        self.num_rep = 64  # RelGAN

        # ===log===
        self.log_time_str = strftime("%m%d_%H%M_%S", localtime())
        self.log_filename = strftime("log/log_%s" % self.log_time_str)
        if os.path.exists(self.log_filename + '.txt'):
            i = 2
            while True:
                if not os.path.exists(self.log_filename + '_%d' % i + '.txt'):
                    log_filename = self.log_filename + '_%d' % i
                    break
                i += 1
        self.log_filename = self.log_filename + '.txt'

        # Automatically choose GPU or CPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            os.system('nvidia-smi -q -d Utilization > gpu')
            with open('gpu', 'r') as _tmpfile:
                self.util_gpu = list(map(int, re.findall(r'Gpu\s+:\s*(\d+)\s*%', _tmpfile.read())))
            os.remove('gpu')
            if len(self.util_gpu):
                self.device = self.util_gpu.index(min(self.util_gpu))
            else:
                self.device = 0
        else:
            self.device = -1
        # device=1
        # print('device: ', device)
        torch.cuda.set_device(self.device)

        # ===Save Model and samples===
        self.save_root = 'save/{}/{}/{}_{}_lt-{}_sl{}_temp{}_T{}/'.format(time.strftime("%Y%m%d"),
                                                                          self.dataset,
                                                                          self.run_model,
                                                                          self.model_type,
                                                                          self.loss_type,
                                                                          self.max_seq_len,
                                                                          self.temperature,
                                                                          self.log_time_str)
        self.save_samples_root = self.save_root + 'samples/'
        self.save_model_root = self.save_root + 'models/'

        self.oracle_state_dict_path = 'pretrain/oracle_data/oracle_lstm.pt'
        self.oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}.pt'
        self.multi_oracle_state_dict_path = 'pretrain/oracle_data/oracle{}_lstm.pt'
        self.multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt'

        self.pretrain_root = 'pretrain/{}/'.format(self.dataset if self.if_real_data else 'oracle_data')
        self.pretrained_gen_path = self.pretrain_root + 'gen_MLE_pretrain_{}_{}_sl{}_sn{}.pt'.format(self.run_model,
                                                                                                     self.model_type,
                                                                                                     self.max_seq_len,
                                                                                                     self.samples_num)
        self.pretrained_dis_path = self.pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}.pt'.format(self.run_model,
                                                                                                 self.model_type,
                                                                                                 self.max_seq_len,
                                                                                                 self.samples_num)
        self.pretrained_clas_path = self.pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}.pt'.format(self.run_model,
                                                                                                   self.model_type,
                                                                                                   self.max_seq_len,
                                                                                                   self.samples_num)
        self.signal_file = 'run_signal.txt'

        self.tips = ''

        assert self.k_label >= 2, 'Error: k_label = {}, which should be >=2!'.format(self.k_label)

        # Create Directory
        dir_list = ['save', 'savefig', 'log', 'pretrain', 'dataset',
                    'pretrain/{}'.format(self.dataset if self.if_real_data else 'oracle_data')]
        if not self.if_test:
            dir_list.extend([self.save_root, self.save_samples_root, self.save_model_root])
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)
