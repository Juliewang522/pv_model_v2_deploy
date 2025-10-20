from data_provider.data_loader import Dataset_Influx_Meteo_Pred
from torch.utils.data import DataLoader
from datetime import datetime

data_dict = {'SolarPred': Dataset_Influx_Meteo_Pred}

def data_provider(args, flag, scalers):
    Data = data_dict[args.data]

    if  flag == 'influx_pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1

    data_set = Data(
        flag=flag, t0=args.t0,
        size=[args.seq_len, args.label_len, args.pred_len],
        scalers=scalers,
        station_id=args.station_id,
        latitude=args.latitude,
        longitude=args.longitude,
        inv_sncodes=args.inv_sncodes
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader

