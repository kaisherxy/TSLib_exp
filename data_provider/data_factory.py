from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Pred
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PV': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'TEST':
        shuffle_flag = False; batch_size = args.batch_size
    elif flag == 'pred':
        # 预测阶段配置：不打乱顺序，保留不完整批次，批次大小为1（单样本预测）
        shuffle_flag = False; batch_size = 1
        Data = Dataset_Pred  # 使用专门设计的预测数据集类
    else:
        shuffle_flag = False; batch_size = args.batch_size
    drop_last = False # 不丢弃不完整批次
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,# 打乱顺序
            num_workers=args.num_workers,
            drop_last=drop_last)# 丢弃不完整批次
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else: # 预测
        if args.data == 'm4': # 短期
            drop_last = False
        elif flag == 'pred':
            # 实例化数据集对象
            data_set = Data(
                root_path=args.root_path,  # 数据集根目录（如'./data/'）
                data_path=args.data_path,  # 数据文件名（如'WTH.csv'）
                flag=flag,  # 数据阶段标记（train/test/val/pred）
                size=[args.seq_len, args.label_len, args.pred_len],  # 时序参数配置
                features=args.features,  # 特征模式（M/S/MS）
                target=args.target,  # 预测目标列名（如'OT'）
                inverse=args.inverse,  # 是否对输出做逆标准化（True/False）
                timeenc=timeenc,  # 是否应用时间编码（0/1）
                freq=freq,  # 时间编码频率（如'h'/'t'等）
                cols=None  # 指定使用的数据列（None表示全选）
            )
        else:
            data_set = Data(
                args = args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
