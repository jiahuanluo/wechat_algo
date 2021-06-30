import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import utils
import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import logging
import sys
import glob
from mmoe import MMOE
from evaluation import evaluate_deepctr

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
parser = argparse.ArgumentParser("mmoe")
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--stage', type=str, required=True, help='stage')
parser.add_argument('--epoch', type=int, default=3, help='epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--embedding_dim', type=int, default=128, help='mbedding_dim')
parser.add_argument('--fraction', type=float, default=1.0, help='fraction')
parser.add_argument('--dnn_hidden_units', type=int, default=256, help='dnn_hidden_units')
parser.add_argument('--expert_dim', type=int, default=8, help='expert_dim')
args = parser.parse_args()
args.name = 'experiments/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*/*.py') + glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)
if __name__ == "__main__":
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    df = pd.read_csv("data/lgb.csv")
    data = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    data = data.sample(frac=args.fraction, replace=False)

    play_cols = ['is_finish', 'play_times', 'play', 'stay']
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    cols = [f for f in data.columns if f not in ['date_'] + play_cols + y_list]
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'weekday']
    dense_features = [f for f in cols if f not in sparse_features]

    # 1.fill nan dense_feature and do simple Transformation for dense features
    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )

    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)

    logging.info('data.shape: {}'.format(data.shape))
    # logging.info('data.columns', data.columns.tolist())
    # logging.info('unique date_: ', data['date_'].unique())

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    if args.stage == 'offline':
        train_model_input = {name: train[name] for name in feature_names}
        train_labels = [train[y].values for y in target]
    else:
        train_model_input = {name: data[name] for name in feature_names}
        train_labels = [data[y].values for y in target]
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=args.expert_dim,
                       dnn_hidden_units=(args.dnn_hidden_units, args.dnn_hidden_units),
                       tasks=['binary', 'binary', 'binary', 'binary'])
    train_model.compile("adagrad", loss='binary_crossentropy')
    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target, logging)

    t1 = time.time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time.time()
    logging.info('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    logging.info('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test[['userid', 'feedid'] + target].to_csv(os.path.join(args.name, 'result.csv'), index=None, float_format='%.6f')
    logging.info('to_csv ok')
