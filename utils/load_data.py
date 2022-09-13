import torch
import numpy as np
import h5py
import scipy.io
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def reorg_data(x,nt=3,nmonth=225):
    """
    reorganize the raw data to use the predictor in month t, (t-1),...,(t-nt-1) as the network inputs for the predictand in month t
    """
    n0,H,W = x.shape
    y = np.full((nmonth,nt,H,W),np.nan)
    for i in range(nt):
        y[:,i] = x[-nmonth-i:n0-i]

    return y


def load_data(args):
    """Return data loader

    Args:
        args: include the data directory, batch size, etc.

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    # # load the DETRENDED raw predictors data: 2002.01-2020.12
    predictors_raw = scipy.io.loadmat(args.data_dir + 'predictors_raw.mat')
    sTWSA_raw = predictors_raw['sTWSA']
    CWSC_raw = predictors_raw['CWSC']
    P_raw = predictors_raw['P']
    T_raw = predictors_raw['T']
    CWB_raw = predictors_raw['CWB']
    CPA_raw = predictors_raw['CPA']
    CCWB_raw = predictors_raw['CCWB']

    # # reorganize the predictor data
    sTWSA = reorg_data(sTWSA_raw,nt=args.nt,nmonth=args.nmonth)
    CWSC = reorg_data(CWSC_raw,nt=args.nt,nmonth=args.nmonth)
    P = reorg_data(P_raw,nt=args.nt,nmonth=args.nmonth)
    T = reorg_data(T_raw,nt=args.nt,nmonth=args.nmonth)
    CWB = reorg_data(CWB_raw,nt=args.nt,nmonth=args.nmonth)
    CPA = reorg_data(CPA_raw,nt=args.nt,nmonth=args.nmonth)
    CCWB = reorg_data(CCWB_raw,nt=args.nt,nmonth=args.nmonth)
    x = np.concatenate((sTWSA,CWSC,P,T,CWB,CPA,CCWB),axis=1)  # shape: N'*(np*nt)*H*W, where np is number of predictors considered

    # # load the DETRENDED GRACE TWSA data: 2002.04-2020.12
    y = scipy.io.loadmat(args.data_dir + 'GRACE_TWSA.mat')
    y = y['GRACE_TWSA']
    y = np.expand_dims(y,axis=1)  # shape: N*1*H*W

    # # load the linear trend of GRACE(-FO) TWSA
    trend = scipy.io.loadmat(args.data_dir + 'trend.mat')
    trend = trend['trend']   # shape: N*H*W

    assert x.shape[0] == y.shape[0], 'The number of input and output sampple must be the same'

    # land mask (1 degree)
    land_mask = scipy.io.loadmat(args.data_dir + 'land_mask.mat')
    land_mask = land_mask['land_mask']

    x[x!=x], y[y!=y] = 0.0, 0.0 # set the nan values as zeros
    x, y = x[:,:,:144], y[:,:,:144]  # consider the region spanning 60S-84N, 180W-180E, i.e., HxW = 144x360
    trend, land_mask = trend[:,:144], land_mask[:144]

    with h5py.File(args.data_dir + "in_output_data_nt{}.hdf5".format(args.nt), 'w') as f:
        f.create_dataset(name='x', data=x, dtype='f', compression='gzip')
        f.create_dataset(name='y', data=y, dtype='f', compression='gzip')

    # # training data
    x_train, y_train = x[:args.ntrain], y[:args.ntrain]
    # # leave out the data in the training and missing months, the remaining data are used for testing
    id_train_gap = np.concatenate((np.arange(args.ntrain),np.arange(183,194)),axis=0)
    x_test, y_test = np.delete(x,id_train_gap,axis=0), np.delete(y,id_train_gap,axis=0)

    print("input of training data shape: {}".format(x_train.shape))
    print("output training data shape: {}".format(y_train.shape))
    print("input of test data shape: {}".format(x_test.shape))
    print("output test data shape: {}".format(y_test.shape))

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    # training data loader
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test data loader
    test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # simple statistics of training output data
    y_train_mean = np.mean(y_train, 0)
    y_train_var = np.sum((y_train - y_train_mean) ** 2)
    train_stats = {}
    train_stats['y_mean'] = y_train_mean
    train_stats['y_var'] = y_train_var
    # simple statistics of test output data
    y_test_mean = np.mean(y_test, 0)
    y_test_var = np.sum((y_test - y_test_mean) ** 2)
    test_stats = {}
    test_stats['y_mean'] = y_test_mean
    test_stats['y_var'] = y_test_var

    # number of input and output channels
    nic, noc = x_test.shape[1], y_test.shape[1]

    return train_loader, test_loader, train_stats, test_stats, nic, noc, trend, land_mask

