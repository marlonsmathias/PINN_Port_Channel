import torch
import numpy as np
#import scipy.io
import argparse
from pinn.neural_net import MLP
from pinn.get_points import normalize
from pinn.util import log
from pathlib import Path
import matplotlib.pyplot as plt
from pinn.evaluate import get_pars, evaluate

def main():

    model_path = Path('model.pt')
    #data_path = args.path
    t = 3600*1.
    x = 0.
    y = 10.

    pars = get_pars(model_path)

    # Define grid
    #t = np.array(t)
    #t = np.array(x)
    #y = np.linspace(pars['yi'],pars['yf'],ny)

    [e_pred,u_pred,v_pred,t_grid,x_grid,y_grid,D,X] = evaluate(t,x,y,model_path)

    print(e_pred)

if __name__== "__main__":
    main()