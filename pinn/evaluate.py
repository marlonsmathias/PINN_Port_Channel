import torch
import numpy as np
from pinn.neural_net import MLP
from pinn.get_points import normalize
from pinn.util import log
from scipy.io import loadmat
from scipy.interpolate import interpn

def get_pars(model_path):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    
    return pars

def get_loss(model_path):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['loss']
    
    return pars

def evaluate(t,x,y,model_path,device='cpu'):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    model = MLP(pars,device)
    model.load_state_dict(model_file['model'])
    model.eval()

    model = model.to(device)

    # Define grid
    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))
    X = torch.tensor(X,dtype=torch.float).to(device)

    # Evaluate model
    Y_pred = model(X)

    Y_pred = Y_pred.cpu().detach().numpy()

    e_pred = Y_pred[:,0].reshape(t_grid.shape)
    u_pred = Y_pred[:,1].reshape(t_grid.shape)
    v_pred = Y_pred[:,2].reshape(t_grid.shape)

    return e_pred, u_pred, v_pred, t_grid, x_grid, y_grid, X

def get_residuals(t,x,y,model_path,device='cpu'):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    model = MLP(pars,device)
    model.load_state_dict(model_file['model'])
    model.eval()

    model = model.to(device)

    # Define grid
    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X_np = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))

    X = torch.tensor(X_np,dtype=torch.float,requires_grad=True).to(device)

    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)

    H = pars['hi'] * (1-x/pars['xf']) + pars['hf'] * x/pars['xf'] - pars['hy'] * torch.abs(y/pars['yf'])

    # Get residuals
    # Forward pass
    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)
    Y = model(torch.hstack((t,x,y)))
    
    e = Y[:,0].reshape(-1, 1)
    u = Y[:,1].reshape(-1, 1)
    v = Y[:,2].reshape(-1, 1)

    # Get derivatives
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    e_y = torch.autograd.grad(e, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
                                
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]

    # Intermediate terms
    d = e + H
    ud = u * d
    vd = v * d

    # Derivative of intermediate terms
    ud_x = torch.autograd.grad(ud, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    vd_y = torch.autograd.grad(vd, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    ud_t = torch.autograd.grad(ud, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    vd_t = torch.autograd.grad(vd, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]

    # Diffusive term
    Am = 0.5 * pars['C'] * pars['dx'] * pars['dy'] * torch.sqrt(torch.square(u_x) + torch.square(v_y) + 0.5*torch.square(u_y + v_x) + np.finfo(float).eps)
    # Adding eps to the equation above is needed to avoid the discontinuity at the derivative of sqrt(X^2), otherwise autograd returns nan

    F1 = 2 * H * Am * u_x
    F2 = 2 * H * Am * u_y
    F3 = H * Am * (u_y + v_x)
    
    F1_x = torch.autograd.grad(F1, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]       
    F2_y = torch.autograd.grad(F2, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    F3_x = torch.autograd.grad(F3, x, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    F3_y = torch.autograd.grad(F3, y, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]

    Fx = F1_x + F3_y
    Fy = F2_y + F3_x

    # Compute residuals
    R1 = (e_t + ud_x + vd_y)/pars['norm_t']
    R2 = (ud_t - Fx + pars['g']*d*e_x)/pars['norm_t']
    R3 = (vd_t - Fy + pars['g']*d*e_y)/pars['norm_t']

    #R1 = e_t
    #R2 = ud_x
    #R3 = vd_y

    R1 = R1.cpu().detach().numpy()
    R1 = R1.reshape(t_grid.shape)
    R2 = R2.cpu().detach().numpy()
    R2 = R2.reshape(t_grid.shape)
    R3 = R3.cpu().detach().numpy()
    R3 = R3.reshape(t_grid.shape)

    return R1, R2, R3, t_grid, x_grid, y_grid, X

def get_volumes(t,nx,ny,model_path,device='cpu'):

    # Load model
    model_file = torch.load(model_path)
    pars = model_file['pars']
    model = MLP(pars,device)
    model.load_state_dict(model_file['model'])
    model.eval()

    model = model.to(device)

    # Define grid
    x = np.linspace(pars['xi'],pars['xf'],nx)
    y = np.linspace(0,pars['yf'],ny)

    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X_np = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))

    X = torch.tensor(X_np,dtype=torch.float,requires_grad=True).to(device)

    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)

    H = pars['hi'] * (1-x/pars['xf']) + pars['hf'] * x/pars['xf'] - pars['hy'] * torch.abs(y/pars['yf'])

    # Get residuals
    # Forward pass
    t = X[:,0].reshape(-1, 1)
    x = X[:,1].reshape(-1, 1)
    y = X[:,2].reshape(-1, 1)
    Y = model(torch.hstack((t,x,y)))
    
    e = Y[:,0].reshape(-1, 1)
    u = Y[:,1].reshape(-1, 1)
    v = Y[:,2].reshape(-1, 1)

    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    
    d = e + H
    ud = u * d

    e_t = e_t.cpu().detach().numpy()
    e_t = e_t.reshape(t_grid.shape)
    ud = ud.cpu().detach().numpy()
    ud = ud.reshape(t_grid.shape)

    inflow = np.sum(ud[:,0,:],axis=1)*pars['yf']/ny
    vol_change = np.sum(np.sum(e_t,axis=2),axis=1)*pars['yf']*pars['xf']/(nx*ny)


    return inflow, vol_change

def load_data(t,x,y,save_path):

    [x_grid, t_grid, y_grid] = np.meshgrid(x,t,y)
    X = np.hstack((t_grid.flatten()[:,None],x_grid.flatten()[:,None],y_grid.flatten()[:,None]))

    data = loadmat(save_path)
    t_data = data['T'].flatten()
    x_data = data['X'].flatten()
    y_data = data['Y'].flatten()

    X_data = (t_data,x_data,y_data)
    E_data = data['E'].transpose((2,0,1))
    U_data = data['U'].transpose((2,0,1))
    V_data = data['V'].transpose((2,0,1))
    
    e_int = interpn(X_data,E_data,X).reshape(t_grid.shape)
    u_int = interpn(X_data,U_data,X).reshape(t_grid.shape)
    v_int = interpn(X_data,V_data,X).reshape(t_grid.shape)

    return e_int, u_int, v_int
    