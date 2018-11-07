anor_test = '../../data/HTTP_2010/anomalousTrafficTest.txt'
nor_test = '../../data/HTTP_2010/normalTrafficTest.txt'
nor_train = '../../data/HTTP_2010/normalTrafficTraining.txt'
request_file = '../../data/HTTP_2010/request.txt'
label_file = '../../data/HTTP_2010/label.txt'
rnnlm_log_file = '../../model/emb200-hid128/rnnlm_log.txt'
ae_log_file = '../../model/emb200-hid128/ae_log.txt'
rnnlm_model_file = "../../model/emb200-hid128/model.params.RNNLM"
ae_model_file = "../../model/emb200-hid128/model.params.AE"
rnnlm_feature_npy = "../../data/emb200-hid128/rnnlm_feature.npy"
ae_feature_npy = "../../data/emb200-hid128/ae_feature.npy"
label_npy = "../../data/emb200-hid128/label.npy"

# model parameters
cuda_num = 2
epoch_num = 20
batch_size = 1
learning_rate = 0.0001

# net parameters
embed_size = 200
num_layers = 1
hidden_size = 128