import os

# data directories
data_root = '../../data/HTTP_2010'
train_fn = '../../data/HTTP_2010/request_train_bpe.txt'
test_fn = '../../data/HTTP_2010/request_test.txt'
train_bpe_fn = "../../data/HTTP_2010/request_train_bpe.txt"
test_bpe_fn = "../../data/HTTP_2010/request_test_bpe.txt"
train_label_fn = "../../data/HTTP_2010/label_train.txt"
test_label_fn = "../../data/HTTP_2010/label_test.txt"

anor_test = '../../data/HTTP_2010/anomalousTrafficTest.txt'
nor_test = '../../data/HTTP_2010/normalTrafficTest.txt'
nor_train = '../../data/HTTP_2010/normalTrafficTraining.txt'
request_file = '../../data/HTTP_2010/request.txt'
label_file = '../../data/HTTP_2010/label.txt'

# model directories
model_root = '../../model/emb200-hid128-bpe2'
rnnlm_log_file = os.path.join(model_root, 'rnnlm_log.txt')
ae_log_file = os.path.join(model_root, 'ae_log.txt')
rnnlm_model_file = os.path.join(model_root, 'model.params.RNNLM')
ae_model_file = os.path.join(model_root, 'model.params.AE')
rnnlm_feature_npy = os.path.join(model_root, 'rnnlm_feature.npy')
ae_feature_npy = os.path.join(model_root, 'ae_feature.npy')
label_npy = os.path.join(model_root, 'label.npy')

# model parameters
cuda_num = 3
epoch_num = 20
batch_size = 64
learning_rate = 0.0001

# net parameters
embed_size = 200
num_layers = 1
hidden_size = 128