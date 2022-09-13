"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.
The codes are modified after Dr. Yinhao Zhu's repository: https://github.com/cics-nd/cnn-surrogate
"""

import torch
import numpy as np
from time import time
from args import args
from models.bayes_nn import BayesNN
from models.svgd import SVGD
from utils.load_data import load_data
from utils.misc import mkdirs, logger
from utils.plot import plot_prediction_bayes, save_stats, plot_R_NSE
import json
import sys
import h5py
import scipy.io


# train the network on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# load data
train_loader, test_loader, train_stats, test_stats, nic, noc, trend, land_mask = load_data(args)
logger['train_output_var'] = train_stats['y_var']
logger['test_output_var'] = test_stats['y_var']
print('Loaded data!')

# deterministic NN
if args.net == 'RRDB':
    # from models.RRDB import Net
    # model = Net(nic, noc, nf=args.features,act_fun=args.act_fun).to(device)
    from RRDBdua import RRDBdua
    model = RRDBdua(nic, noc, nf=args.features,act_fun=args.act_fun).to(device)
else:
    from models.CBAM import Net
    model = Net(nic, noc, nf=args.features,act_fun=args.act_fun).to(device)
print(model)
# Bayesian NN
bayes_nn = BayesNN(model, n_samples=args.n_samples).to(device)

# Initialize SVGD
svgd = SVGD(bayes_nn, train_loader)


def test(epoch, logger):
    """Evaluate model during training. 
    Print predictions including 4 rows:
        1. target
        2. predictive mean
        3. error of the above two
        4. two sigma of predictive variance

    Args:
        test_fixed (Tensor): (2, N, *), `test_fixed[0]` is the fixed test input, 
            `test_fixed[1]` is the corresponding target
    """
    bayes_nn.eval()
    
    mse_test, nlp_test = 0., 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        mse, nlp, output = bayes_nn._compute_mse_nlp(input, target, 
                            size_average=True, out=True)
        # output: S x N x oC x oH x oW --> N x oC x oH x oW
        y_pred_mean = output.mean(0)        
        EyyT = (output ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        y_noise_var = (- bayes_nn.log_beta).exp().mean()
        y_pred_var =  EyyT - EyEyT + y_noise_var

        mse_test += mse.item()
        nlp_test += nlp.item()

        if batch_idx == len(test_loader) - 1 and epoch % args.plot_freq == 0:
            n_samples = 2  # number of test samples to be plotted
            idx = torch.randperm(input.size(0))[: n_samples]
            samples_pred_mean = y_pred_mean[idx].cpu().numpy()
            samples_target = target[idx].cpu().numpy()
            samples_pred_var = y_pred_var[idx].cpu().numpy()
           
            for i, index in enumerate(idx):
                print('epoch {}: plotting {}-th prediction'.format(epoch, index))
                plot_prediction_bayes(args.pred_dir, samples_target[i], land_mask,
                    samples_pred_mean[i], samples_pred_var[i], epoch, index)

    rmse_test = np.sqrt(mse_test / len(test_loader))
    r2_test = 1 - mse_test * target.numel() / logger['test_output_var']
    mnlp_test = nlp_test / len(test_loader)    
    logger['rmse_test'].append(rmse_test)
    logger['r2_test'].append(r2_test)
    logger['mnlp_test'].append(mnlp_test)
    print("epoch {}, testing  r2: {:.4f}, test mnlp: {}".format(
        epoch, r2_test, mnlp_test))


def eva_allsamples(args,trend,land_mask):
    with h5py.File(args.data_dir+'in_output_data_nt{}.hdf5'.format(args.nt), 'r') as f:
        x = f['x'][()]
        y = f['y'][()]
    y = np.squeeze(y)  # N*1*H*W --> N*H*W

    y_pred, y_pred_var = pred_samples(x,land_mask)
    y, y_pred = y+trend, y_pred+trend
    scipy.io.savemat(args.pred_dir +'/y_ypred_var.mat',dict(y=np.squeeze(y),
                                                            y_pred=np.squeeze(y_pred),
                                                            y_pred_var=np.squeeze(y_pred_var)))
    # remove the training and missing samples to calculate the R and NSE metrics based on the testing data
    id_train_gap = np.concatenate((np.arange(args.ntrain),np.arange(183,194)),axis=0)
    ytest, y_test_pred = np.delete(y,id_train_gap,axis=0), np.delete(y_pred,id_train_gap,axis=0)

    plot_R_NSE(ytest, y_test_pred,land_mask,args.pred_dir)


def pred_samples(x,land_mask):
    bayes_nn.eval()
    n,_,H,W = x.shape
    x[x!=x] = 0.0 # set the nan values as zeros
    y_pred, y_pred_var = np.full((n,H,W),np.nan), np.full((n,H,W),np.nan)
    for i in range(x.shape[0]):
        x_tensor = (torch.FloatTensor(x[[i]])).to(device)
        y_hat, pred_var = bayes_nn.predict(x_tensor)
        y_hat, pred_var = y_hat.data.cpu().numpy(), pred_var.data.cpu().numpy()
        y_hat,pred_var = np.squeeze(y_hat), np.squeeze(pred_var)
        y_hat[land_mask!=land_mask], pred_var[land_mask!=land_mask] = np.nan, np.nan
        y_pred[i], y_pred_var[i] = y_hat, pred_var

    return y_pred, y_pred_var


if args.pre_trained:
    # post-processing using the pretrained network ## #
    bayes_nn.load_state_dict(torch.load(args.ckpt_dir + '/model_epoch{}.pth'.format(args.epochs)))
    test(args.epochs, logger)
    eva_allsamples(args,trend,land_mask)
    sys.exit(0)


print('Start training.........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    svgd.train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
training_time = time() - tic
print('Finished training:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
    .format(args.epochs, args.ntrain, args.n_samples, training_time))

# compute and plot the accuracy metrics
eva_allsamples(args,trend,land_mask)

# save training results
x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
# plot the rmse, r2-score curve and save them in txt
save_stats(args.train_dir, logger, x_axis)

args.training_time = training_time
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
