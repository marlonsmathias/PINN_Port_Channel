import torch
import numpy as np
#import scipy.io
import argparse
import time
from pinn.neural_net import PINN
from pinn.util import log
import pinn.get_points
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        prog='Training step',
        usage='%(prog)s [options] parser',
        description='Parser for hyperparams training')
    
    parser.add_argument('--path',
                        type=str,
                        default='',
                        help='Use to manually select the model file name')

    parser.add_argument('--comment',
                        type=str,
                        default='',
                        help='String to be added to the end of the automatically generated file name')

    parser.add_argument('--folder',
                        type=str,
                        default='models',
                        help='Folder where the automatically named model will be saved')   

    parser.add_argument('--resume',
                        type=str,
                        default='',
                        help='Model to be used as initial guess')
    
    parser.add_argument('--nb',
                        type=int,
                        default=1000,
                        help='Number of boundary points')

    parser.add_argument('--nf',
                        type=int,
                        default=10000,
                        help='Number of function evaluation points')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=50000,
                        help='Number of epochs for training')

    parser.add_argument('--fn',
                        type=str,
                        default='1-0-0',
                        help='Fourier nodes for inputs t, x and y')

    parser.add_argument('--fs',
                        type=str,
                        default='0-1-1',
                        help='Fourier nodes sigma for inputs x and y')

    parser.add_argument('--nf_delta',
                        type=int,
                        default=0,
                        help='Value of nf used to compute dx*dy for the Smagorinsky Diffusivity. 0 -> use the same as actual nf')

    parser.add_argument('--nlayers',
                        type=int,
                        default=2,
                        help='MLP number residual blocks')

    parser.add_argument('--nneurons',
                        type=int,
                        default=20,
                        help='MLP number neurons per layer')

    parser.add_argument('--shuffle',
                        type=int,
                        default=0,
                        help='Reshuffle sample every n itertions - 0 for fixed sample')

    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed')                

    parser.add_argument('--opt_method',
                        type=str,
                        default='adam',
                        help='Optimization algorithm (adam, lbfgs or sgd)')

    parser.add_argument('--opt_lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for the optimization algorithm')

    parser.add_argument('--loss',
                        type=str,
                        default='l1',
                        help='Type of reduction to be used for each loss (l1 or mse)')

    parser.add_argument('--dev',
                        type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to run the model')
    
    args = parser.parse_args()
    
    return args

def main():

    # Define parameters
    pars = dict()
    pars['xi'] = 0
    pars['xf'] = 10000
    pars['yf'] = 250
    pars['tf'] = 12*3600
    pars['Amplitude'] = 1
    pars['hi'] = 15
    pars['hf'] = 5
    pars['hy'] = 2
    pars['g'] = 9.81
    pars['C'] = 0.2

    # Retrive arguments
    args = get_args()
    nb = args.nb
    nf = args.nf
    pars['epochs'] = args.epochs
    pars['shuffle'] = args.shuffle
    device = args.dev
    resume = args.resume

    nf_delta = args.nf_delta
    if nf_delta == 0:
        nf_delta = nf
        nf_delta_text = ''
    else:
        nf_delta_text = f'nf_delta{nf_delta}'
    

    pars['opt_method'] = args.opt_method
    pars['opt_lr'] = args.opt_lr
    pars['loss_type'] = args.loss

    pars['layers'] = [args.nneurons for i in range(0,args.nlayers)]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.path) == 0:
        pars['save_path'] = Path(f'{args.folder}/model_nf{nf}_nb{nb}{nf_delta_text}_fn{args.fn}_fs{args.fs}_MLPRes_2x{args.nlayers}x{args.nneurons}_shuffle{args.shuffle}_seed{args.seed}_{args.opt_method}_lr{args.opt_lr}_loss_{args.loss}{args.comment}.pt')
    else:
        pars['save_path'] = Path(args.path)

    if pars['save_path'].is_file():
        return

    pars['fourier_nodes'] = [int(i) for i in args.fn.split('-')] # for t, x and y
    pars['fourier_sigma'] = [float(i) for i in args.fs.split('-')] # for t, x and y

    log.info(f'Model will be saved to: {pars["save_path"]}')
    log.info(f'Number of samples - Boundary conditions: {nb}, Function evaluation: {nf}')
    log.info(f'Using device: {device}')

    # Get normalization parameters and mean mesh spacing
    pars['norm_t'] = 1./pars['tf']
    pars['norm_x'] = 1./(pars['xf'] - pars['xi'])
    pars['norm_y'] = 1./pars['yf']
    pars['dx'] = 1./(pars['norm_x']*nf**(1/3))
    pars['dy'] = 1./(pars['norm_y']*nf**(1/3))

    # Initialize fourier layer coefficients
    pars['B_fourier'] = pinn.get_points.initialize_fourier_feature_network(pars['fourier_nodes'],pars['fourier_sigma'])

    # Train model
    model = PINN(nb, nf, pars, device)
    if len(resume) != 0:
        resume_file = torch.load(resume)
        model.net.load_state_dict(resume_file['model'])

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time

    log.info(f'Training time: {elapsed:.4f}s')
    model.save(-1)
    log.info('Finished training.')

if __name__== "__main__":
    main()