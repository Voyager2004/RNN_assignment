class Config:
    def __init__(self):
        self.data_dir = './data/'
        self.data_path = self.data_dir + 'peot.txt'
        self.pickle_path = self.data_dir + 'tang.npz'
        self.load_path = './checkpoints/peot.pt'
        self.save_path = './checkpoints/peot.pt'

        self.do_train = True
        self.do_test = True
        self.do_predict = True
        self.do_load_model = False

        self.num_epoch = 20
        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.max_gen_len = 200
        self.max_len = 125
        # 使用权重共享和 GRU 优化模型规模
        # 为了便于权重共享，embedding_dim 与 hidden_dim 设为一致
        self.embedding_dim = 512
        self.hidden_dim = 512
