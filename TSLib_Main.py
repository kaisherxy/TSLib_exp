# https://github.com/thuml/Time-Series-Library/tree/main
import argparse
import os, sys
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from scipy.io import loadmat, savemat
import random
import numpy as np
import time
import re
from utils.metrics import metric
import io

# 记录开始时间
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
start_time = time.time()
print(f"程序开始执行，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("当前 Python 解释器路径：", sys.executable)

if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"PyTorch版本: {torch.__version__}, CUDA版本: {torch.version.cuda}")
else:
    print("警告: 未检测到GPU，模型将在CPU上运行，训练速度会较慢")

results_path = 'results/PV_results.mat'

model_name = 'iTransformer'

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TSLib')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # short_term_forecast针对m4数据集有较多定制化逻辑，代码灵活性和对特定数据集适配性更强； long_term_forecast相对更通用和简洁，未做过多针对特定数据集的特殊处理

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')

    parser.add_argument('--model', type=str, default=model_name,
                        help='model name, options: [Autoformer, Transformer, FEDformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='PV', help='data')
    parser.add_argument('--root_path', type=str, default='data/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='PV_data.csv', help='data file')

    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    parser.add_argument('--target', type=str, default='Power', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')# only for M4
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True) # 输出逆归一化

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')# 插值任务

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')

    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')# 默认512
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')# 默认 8. 多头注意力
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')# 默认2. 编码器层数
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')# 默认1. 解码器层数
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')# 默认值2048
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor') # 默认为1
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')# 默认0.1
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN') # 默认96

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')# num_workers设置过大时，会导致运行慢，频繁分离.默认10
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')# 默认10
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')# 默认 32
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience') # 默认 3
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')# 默认 0.0001
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps. MPS:Apple芯片的 GPU 加速技术
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw) 使用动态时间规整（DTW）度量标准的控制器（DTW 计算耗时较长，除非必要否则不建议使用）
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # 深度学习模型的训练和测试流程，支持多轮实验，并根据参数配置动态生成实验名称
    if args.is_training: # 根据args.is_training参数决定执行训练 + 测试或仅测试
        # 训练分支（args.is_training为 True）：执行多轮训练和测试
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name, args.model_id, args.model,
                args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model,
                args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
                args.factor, args.embed, args.distil, args.des, ii)

            print('\n>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            y_test_predict_norm, test_y_label = exp.test(setting)

            print('\n>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            y_pre_predict_norm, y_label_predict = exp.predict(setting, True)

            # 支持 GPU 内存管理（MPS 和 CUDA）
            if args.gpu_type == 'mps': # Apple Silicon 芯片（如 M1、M2 系列）的 GPU 加速技术
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()


    else: # 测试分支（args.is_training为 False）：仅执行单轮测试
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model,
            args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model,
            args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
            args.factor, args.embed, args.distil, args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        y_test_predict_norm, test_y_label = exp.test(setting, test=1)

        print('\n>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        y_pre_predict_norm, y_label_predict = exp.predict(setting, True)

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


y_test_predict_norm = y_test_predict_norm.squeeze(axis=-1)  # 形状：(总样本数, 12)
test_y_label = test_y_label.squeeze(axis=-1)  # 形状：(总样本数, 12)

y_pre_predict_norm = y_pre_predict_norm.squeeze(axis=-1)  # 形状：(总样本数, 12)
y_label_predict = y_label_predict.squeeze(axis=-1)  # 形状：(总样本数, 12)

# 打印完成信息
print("\n=== 训练和预测完成 ===")
print("Future 12-step Prediction:\n", y_pre_predict_norm)
print("Future 12-step 真实值:\n", y_label_predict)

# 构建结果字典，包含预测结果和模型参数
results = {
    'y_test_predict_norm': y_test_predict_norm,
    'test_y_label': test_y_label,
    'y_pre_predict_norm': y_pre_predict_norm,
    'y_label_predict': y_label_predict,
    'model_name': args.model,
}

# 保存结果到MAT文件
try:
    savemat(results_path, {'results': results})
    print(f"结果保存成功: {results_path}")
except Exception as e:
    print(f"保存结果失败: {e}")

# 打印完成信息
mae, mse, rmse, mape, mspe = metric(y_test_predict_norm, test_y_label)
print('测试集指标: rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))

# 保存计时结束
save_end = time.time()
print(f"{args.model}_Python单次总耗时: {save_end - start_time :.6f} 秒")