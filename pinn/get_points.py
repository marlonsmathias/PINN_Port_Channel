import numpy as np

def boundary(n,pars):
    n_inlet = int(np.ceil(n/4))
    n_side = int(np.ceil(n/4))
    n_end = n - n_inlet - 2*n_side

    # Inlet
    t_inlet = np.random.uniform(low=0,high=pars['tf'],size=(n_inlet,1))
    x_inlet = np.zeros((n_inlet,1))
    y_inlet = np.random.uniform(low=0,high=pars['yf'],size=(n_inlet,1))

    X_inlet = np.hstack((t_inlet,x_inlet,y_inlet))
    Y_inlet = np.zeros((n_inlet,3))

    T_inlet = np.hstack((np.zeros((n_inlet,1)), np.ones((n_inlet,2)))) # e is Dirichlet, u and v are Neumann in x

    # End
    t_end = np.random.uniform(low=0,high=pars['tf'],size=(n_end,1))
    x_end = pars['xf'] * np.ones((n_end,1))
    y_end = np.random.uniform(low=0,high=pars['yf'],size=(n_end,1))

    X_end = np.hstack((t_end,x_end,y_end))
    Y_end = np.zeros((n_end,3))

    T_end = np.hstack((np.ones((n_end,1)), np.zeros((n_end,2)))) # e is Neumann in x, u and v are Dirichlet

    # Sides
    t_side1 = np.random.uniform(low=0,high=pars['tf'],size=(n_side,1))
    x_side1 = np.random.uniform(low=pars['xi'],high=pars['xf'],size=(n_side,1))
    y_side1 = np.zeros((n_side,1))
    T_side1 = np.hstack((2*np.ones((n_side,2)),np.zeros((n_side,1)))) # Symmetry: e and u are Neumann in y and v is Dirichlet

    t_side2 = np.random.uniform(low=0,high=pars['tf'],size=(n_side,1))
    x_side2 = np.random.uniform(low=pars['xi'],high=pars['xf'],size=(n_side,1))
    y_side2 = pars['yf'] * np.ones((n_side,1))
    T_side2 = np.hstack((2*np.ones((n_side,1)), np.zeros((n_side,2)))) # e is Neumann in y, u and v are Dirichlet

    X_side1 = np.hstack((t_side1,x_side1,y_side1))
    X_side2 = np.hstack((t_side2,x_side2,y_side2))

    X_side = np.vstack((X_side1,X_side2))
    Y_side = np.zeros((2*n_side,3))

    T_side = np.vstack((T_side1,T_side2)) # e is Neumann in y, u and v are Dirichlet

    # Join all boundaries
    X = np.vstack((X_inlet,X_end,X_side))
    Y = np.vstack((Y_inlet,Y_end,Y_side))
    T = np.vstack((T_inlet,T_end,T_side))

    return X, Y, T

def domain(n,pars):

    t = np.random.uniform(low=0,high=pars['tf'],size=(n,1))
    x = np.random.uniform(low=pars['xi'],high=pars['xf'],size=(n,1))
    y = np.random.uniform(low=0,high=pars['yf'],size=(n,1))

    X = np.hstack((t,x,y))

    return X

def initialize_fourier_feature_network(n_transforms,sigma):
    B = {}
    #B['t'] = np.random.normal(scale=2*np.pi*sigma[0],size=n_transforms[0])
    B['t'] = np.array([2*np.pi*i for i in range(1,n_transforms[0]+1)])
    B['x'] = np.random.normal(scale=2*np.pi*sigma[1],size=n_transforms[1])
    B['y'] = np.random.normal(scale=2*np.pi*sigma[2],size=n_transforms[2])

    return B