import matplotlib.pyplot as plt
import numpy as np
from .misc import to_numpy
import scipy.io
plt.switch_backend('agg')


def save_stats(save_dir, logger, x_axis):

    rmse_train = logger['rmse_train']
    rmse_test = logger['rmse_test']
    r2_train = logger['r2_train']
    r2_test = logger['r2_test']

    if 'mnlp_test' in logger.keys():
        mnlp_test = logger['mnlp_test']
        if len(mnlp_test) > 0:
            plt.figure()
            plt.plot(x_axis, mnlp_test, label="Test: {:.3f}".format(np.mean(mnlp_test[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('MNLP')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/mnlp_test.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/mnlp_test.txt", mnlp_test)
    
    if 'log_beta' in logger.keys():
        log_beta = logger['log_beta']
        if len(log_beta) > 0:
            plt.figure()
            plt.plot(x_axis, log_beta, label="Test: {:.3f}".format(np.mean(log_beta[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('Log-Beta (noise precision)')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/log_beta.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/log_beta.txt", log_beta)

    plt.figure()
    plt.plot(x_axis, r2_train, label="Train: {:.3f}".format(np.mean(r2_train[-5:])))
    plt.plot(x_axis, r2_test, label="Test: {:.3f}".format(np.mean(r2_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'$R^2$-score')
    plt.legend(loc='lower right')
    plt.savefig(save_dir + "/r2.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/r2_train.txt", r2_train)
    np.savetxt(save_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x_axis, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5:])))
    plt.plot(x_axis, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.savefig(save_dir + "/rmse.pdf", dpi=600)
    plt.close()
    np.savetxt(save_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(save_dir + "/rmse_test.txt", rmse_test)


def plot_prediction_bayes(save_dir, target, land_mask, pred_mean, pred_var, epoch, index):
    """Plot predictions at *one* test input

    Args:
        save_dir: directory to save predictions
        target (np.ndarray or torch.Tensor): (1,H,W)
        pred_mean (np.ndarray or torch.Tensor): (1,H,W)
        pred_var (np.ndarray or torch.Tensor): (1,H,W)
        epoch (int): which epoch
        index (int): i-th prediction
    """
    target, pred_mean, pred_var = to_numpy(target[0]), to_numpy(pred_mean[0]), to_numpy(pred_var[0])
    pred_error = target - pred_mean
    sigma = np.sqrt(pred_var)
    print(target.shape, pred_mean.shape, pred_var.shape)

    v_max, v_min, e_max, e_min = 30, -30, 10, -10

    target[land_mask!=land_mask] = np.nan
    pred_mean[land_mask!=land_mask] = np.nan
    pred_error[land_mask!=land_mask] = np.nan
    sigma[land_mask!=land_mask] = np.nan

    fs = 9
    fig, axes = plt.subplots(1, 4, figsize=(12,3))
    # plt.subplots_adjust(hspace=0.6, wspace=0.26)
    for j, ax in enumerate(fig.axes):
        print(j)
        if j == 0:
            cax = ax.imshow(target, cmap='jet', origin='lower',vmin=v_min, vmax=v_max)
            ax.set_title('GRACE TWSA',fontsize=fs)
        elif j == 1:
            cax = ax.imshow(pred_mean, cmap='jet', origin='lower',vmin=v_min, vmax=v_max)
            ax.set_title('CNN TWSA',fontsize=fs)
        elif j == 2:
            cax = ax.imshow(pred_error, cmap='jet', origin='lower',vmin=e_min, vmax=e_max)
            ax.set_title('Error',fontsize=fs)
        else:
            cax = ax.imshow(sigma, cmap='jet', origin='lower')
            ax.set_title('Standard deviation',fontsize=fs)
        cbar = plt.colorbar(cax, ax=ax,fraction=0.02, pad=0.015)
        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.tick_params(labelsize=fs-2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')

    plt.savefig(save_dir + '/epoch_{}_output_{}.png'.format(epoch,index), bbox_inches='tight',dpi=400)
    plt.close(fig)
    print("epoch {}, done with printing sample output {}".format(epoch, index))


def plot_R_NSE(target, output, land_mask, save_dir):
    H, W = target.shape[1], target.shape[2]
    R = np.full((H,W),np.nan)
    NSE = np.full((H,W),np.nan)
    for i in range(H):
        for j in range(W):
            if np.isnan(land_mask[i,j]):
                continue
            else:
                y_target = target[:,i,j]
                y_pred   = output[:,i,j]
                R[i,j] = np.corrcoef(y_target, y_pred)[0,1]
                NSE[i,j] = 1-(np.sum((y_pred-y_target)**2)/np.sum((y_target-np.mean(y_target))**2))

    fs = 12
    fig, axes = plt.subplots(1, 2, figsize=(6,7))
    for j, ax in enumerate(fig.axes):
        print(j)
        if j == 0:
            cax = ax.imshow(R, cmap='jet', origin='lower', vmin=0, vmax=1)
            cbar = plt.colorbar(cax, ax=ax,fraction=0.024, pad=0.015,extend='min')
            ax.set_title('$R$',fontsize=fs)
        else:
            cax = ax.imshow(NSE, cmap='jet', origin='lower', vmin=0, vmax=1)
            cbar = plt.colorbar(cax, ax=ax,fraction=0.024, pad=0.015,extend='min')
            ax.set_title('NSE',fontsize=fs)

        cbar.ax.tick_params(axis='both', which='both', length=0)
        cbar.ax.tick_params(labelsize=fs-2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')

    plt.savefig(save_dir + '/R_NSE.png', bbox_inches='tight',dpi=400)
    plt.close(fig)

    scipy.io.savemat(save_dir +'/R_NSE.mat', dict(R=R,NSE=NSE))

