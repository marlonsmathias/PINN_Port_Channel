#from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pinn.util import log
from pathlib import Path
import pinn.get_points
import time

class MLP(nn.Module):
    
    # Define the MLP

    def __init__(
        self, pars, device
    ) -> None:

        super().__init__()

        # Add number of MLP input and outputs to the layers list
        layers = [2*sum(pars['fourier_nodes'])+2,*pars['layers'],3]

        # Retrieve amplitude parameter
        self.Amplitude = pars['Amplitude']

        # Send Fourier Feature coefficients and normalization parameters to device
        self.B_t = torch.tensor(pars['B_fourier']['t'].reshape(1,-1),dtype=torch.float).to(device)
        self.B_x = torch.tensor(pars['B_fourier']['x'].reshape(1,-1),dtype=torch.float).to(device)
        self.B_y = torch.tensor(pars['B_fourier']['y'].reshape(1,-1),dtype=torch.float).to(device)
        self.norm = torch.tensor([pars['norm_t'],pars['norm_x'],pars['norm_y']],dtype=torch.float).to(device)
        
        # Built the MLP
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(ResidualBlock(_out))
        
        # Remove last block
        modules.pop()

        self.model = nn.Sequential(*modules)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Forward pass
        X_norm = self.normalize(X)
        X_fourier = self.apply_fourier_feature(X_norm)
        Y_n = self.model(X_fourier)
        Y_p = self.particular_solution(X_norm)
        D = self.boundary_distance(X_norm)

        return D * Y_n + Y_p

    def particular_solution(self,X):
        t = X[:,0].reshape(-1, 1)

        e = self.Amplitude * torch.sin(2*np.pi*t)
        uv = torch.zeros_like(t)

        return torch.hstack((e,uv,uv))

    def boundary_distance(self,X):

        #alpha = 26.4 # Reaches 0.99 at 10% of the domain
        alpha = 10.56 # Reaches 0.99 at 25% of the domain

        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)

        dxi = torch.tanh(x*alpha)
        dxf = torch.tanh((1-x)*alpha)
        dyi = torch.tanh(y*alpha)
        dyf = torch.tanh((1-y)*alpha)

        return torch.hstack((dxi,dxf*dyf,dxf*dyi*dyf))

    def normalize(self,X):
        return X*self.norm

    def apply_fourier_feature(self,X):
        cos_t = torch.cos(X[:,0].reshape(-1,1)*self.B_t)
        sin_t = torch.sin(X[:,0].reshape(-1,1)*self.B_t)
        cos_x = torch.cos(X[:,1].reshape(-1,1)*self.B_x)
        sin_x = torch.sin(X[:,1].reshape(-1,1)*self.B_x)
        cos_y = torch.cos(X[:,2].reshape(-1,1)*self.B_y)
        sin_y = torch.sin(X[:,2].reshape(-1,1)*self.B_y)

        return torch.hstack((cos_t,sin_t,X[:,1].reshape(-1,1),cos_x,sin_x,X[:,2].reshape(-1,1),cos_y,sin_y))

class ResidualBlock(nn.Module):

    # Define a block with two layers and a residual connection
    def __init__(self,_size:int):
        super().__init__()
        self.Layer1 = nn.Tanh()
        self.Linear = nn.Linear(_size, _size)
        self.Layer2 = nn.Tanh()

    def forward(self,x):
        return x + self.Layer2(self.Linear(self.Layer1(x)))

class PINN:
    def __init__(self, nb, nf, pars: dict, device: torch.device = 'cpu') -> None:

        # Parameters
        self.pars = pars
        self.device = device
        self.nb = nb
        self.nf = nf

        # Sample points
        self.sample_points()

        # Initialize Network
        self.net = MLP(pars,device)
        self.net = self.net.to(device)

        if pars['loss_type'] == 'l1':
            self.loss = nn.L1Loss().to(device)
        elif pars['loss_type'] == 'mse':
            self.loss = nn.MSELoss().to(device)

        self.min_ls_tol = 0.01
        self.min_ls_wait = 10000
        self.min_ls_window = 1000

        self.start_time = time.time()

        self.ls = 0
        self.iter = 0

        self.ls_hist = np.zeros((pars['epochs'],5))

        # Optimizer parameters
        if pars['opt_method'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=pars['opt_lr'])
        elif pars['opt_method'] == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.net.parameters(),lr=pars['opt_lr'])
        elif pars['opt_method'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=pars['opt_lr'])
        else:
            raise Exception("Unknown optimization method")

    def sample_points(self):

        [X_b, Y_b, T_b] = pinn.get_points.boundary(self.nb, self.pars)
        X_f = pinn.get_points.domain(self.nf,self.pars)

        # Transform to tensors

        # Points from the boundary condition
        self.X_b = torch.tensor(X_b,dtype=torch.float,requires_grad=True).to(self.device)
        self.Y_b = torch.tensor(Y_b,dtype=torch.float,requires_grad=False).to(self.device)

        # Masks for Dirichlet and Neumann conditions
        self.M_Nx = torch.tensor(T_b==1).to(self.device)
        self.M_Ny = torch.tensor(T_b==2).to(self.device)

        # Points for function evaluation
        self.X_f = torch.tensor(X_f,dtype=torch.float,requires_grad=True).to(self.device)
        self.zeros = torch.zeros(self.X_f.shape).to(self.device)

        # Parameters for height
        self.hi = self.pars['hi']
        self.hx = (self.pars['hf']-self.pars['hi'])*self.pars['norm_x']
        self.hy = self.pars['hy']*self.pars['norm_y']

    def NS_loss(self, X: torch.Tensor):

        # Forward pass
        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)
        Y = self.net(torch.hstack((t,x,y)))
        
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

        H = self.hi + x*self.hx - y*self.hy

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
        Am = self.pars['C'] * self.pars['dx'] * self.pars['dy'] * torch.sqrt(torch.square(u_x) + torch.square(v_y) + 0.5*torch.square(u_y + v_x) + np.finfo(float).eps)
        # Adding eps to the equation above is needed to avoid the discontinuity at the derivative of sqrt(x^2), otherwise autograd returns nan

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
        R1 = (e_t + ud_x + vd_y)
        R2 = (ud_t - Fx + self.pars['g']*d*e_x)
        R3 = (vd_t - Fy + self.pars['g']*d*e_y)

        self.ls_f1 = self.loss(R1,torch.zeros_like(R1))
        self.ls_f2 = self.loss(R2,torch.zeros_like(R1))
        self.ls_f3 = self.loss(R3,torch.zeros_like(R1))

        return self.ls_f1 + self.ls_f2 + self.ls_f3

    def boundary_loss(self, X: torch.Tensor, Y_gt: torch.Tensor):

        # Compute the residuals of unmet Neumann boundary conditions 

        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)
        Y = self.net(torch.hstack((t,x,y)))

        Y_Nx0 = torch.autograd.grad(Y[:,0], x, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]
        Y_Nx1 = torch.autograd.grad(Y[:,1], x, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]
        Y_Nx2 = torch.autograd.grad(Y[:,2], x, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]
        Y_Ny0 = torch.autograd.grad(Y[:,0], y, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]
        Y_Ny1 = torch.autograd.grad(Y[:,1], y, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]
        Y_Ny2 = torch.autograd.grad(Y[:,2], y, grad_outputs=torch.ones_like(Y[:,0]),
                                  retain_graph=True, create_graph=True)[0]

        Y_Nx = torch.hstack((Y_Nx0,Y_Nx1,Y_Nx2))
        Y_Ny = torch.hstack((Y_Ny0,Y_Ny1,Y_Ny2))

        Y_pred = Y_Nx*self.M_Nx + Y_Ny*self.M_Ny

        return self.loss(Y_pred,Y_gt)

    def closure(self) -> torch.nn:
        
        self.ls_b = self.boundary_loss(self.X_b,self.Y_b)
        self.ls_f = self.NS_loss(self.X_f)

        self.ls = self.ls_b + self.ls_f

        self.optimizer.zero_grad()
        self.ls.backward()

        return self.ls

    def stopping(self):
        # Stop the training if the median loss of the last min_ls_window steps is not improved in min_ls_wait by a factor of min_ls_tol
        if self.iter > self.min_ls_wait + self.min_ls_window:

            old_list = sorted(self.ls_hist[self.iter-self.min_ls_wait-self.min_ls_window+1:self.iter-self.min_ls_wait,0])
            new_list = sorted(self.ls_hist[self.iter-self.min_ls_window+1:self.iter,0])
            median_ind = self.min_ls_window//2

            if new_list[median_ind] > old_list[median_ind] * (1-self.min_ls_tol):
                return True

        return False

    def train(self):
        self.net.train()

        for i in range(1,self.pars['epochs']):
            self.iter += 1

            if self.pars['shuffle'] and i%self.pars['shuffle']==0:
                self.sample_points()

            try:
                self.optimizer.step(self.closure)
            except KeyboardInterrupt:
                print("Stopped by user")
                self.save(0)
                try:
                    input('Press Enter to resume or Ctrl+C again to stop')
                except KeyboardInterrupt:
                    break
                

            log.info(f'Epoch: {self.iter}, Loss: {self.ls:.3e}, Loss_F: {self.ls_f:.3e} ({self.ls_f1:.3e} + {self.ls_f2:.3e} + {self.ls_f3:.3e}), Loss_B: {self.ls_b:.3e}')

            self.ls_hist[i,:] = torch.hstack((self.ls,self.ls_f1,self.ls_f2,self.ls_f3,self.ls_b)).cpu().detach().numpy()

            # Save the model every n steps
            #if i%10000==0:
            #    self.save(i)

            # Stop if early stopping criterium is met
            #if self.stopping():
            #    break

    def save(self,iter):
        Path(str(self.pars['save_path'].parents[0])).mkdir(parents=True, exist_ok=True)
        if iter == -1:
            save_path = self.pars['save_path']
        elif iter == 0:
            save_path = "{0}_partial.{1}".format(*str(self.pars['save_path']).rsplit('.', 1))
        else:
            save_path = f"{str(self.pars['save_path'])[0:-3]}_iter{iter}.pt"

        log.info(f'Saving model to {save_path}')

        ls_hist_temp = self.ls_hist[0:np.nonzero(self.ls_hist[:,0])[0][-1],:]

        torch.save({'model': self.net.state_dict(),'pars':self.pars,'loss':ls_hist_temp, 'time':time.time()-self.start_time, 'memory':torch.cuda.max_memory_allocated(self.device)/(1024*1024)}, save_path)