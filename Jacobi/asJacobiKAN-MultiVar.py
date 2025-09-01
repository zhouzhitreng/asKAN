#!/usr/bin/env python
# coding: utf-8

# 360 parameters
# total_params = sum(p.numel() for p in Jacobi_model.parameters() if p.requires_grad)
# total_params
# In[1]:
# 

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from LBFGS import *
from JacobiKANLayer import JacobiKANLayer
import copy
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator, AutoMinorLocator
# In[2]:


# Define target function
def target_function(x):

    y = torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]]))))

    return y


# In[3]:


# Define MLP and JacobiKAN
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.layers(x)


class JacobiKAN(nn.Module):
    def __init__(self):
        super(JacobiKAN, self).__init__()
        self.Jacobikan1 = JacobiKANLayer(2, 30, 3,3,3)
        self.Jacobikan2 = JacobiKANLayer(30, 1, 3,3,3)

    def forward(self, x):
        x = self.Jacobikan1(x)
        x = self.Jacobikan2(x)
        return x


# %% active subspace
def get_direction(model, dataset):
    xac = dataset['train_input']
    x_test = dataset['test_input']
    yac = dataset['train_label']
    xac.requires_grad = True
    ymodel = model(xac)
    deri_xac = torch.autograd.grad(ymodel, xac, grad_outputs=torch.ones_like(ymodel), create_graph=True, )
    num1 = len(xac[0,:])
    nac  = len(xac[:,0])
    corall = torch.zeros((num1, num1))
    for i1 in range(nac):
        arr1_col = deri_xac[0][i1,:]
        vector_col = arr1_col.unsqueeze(1)
        arr1_row = arr1_col.T    
        cor = vector_col @ vector_col.T
        if i1 == 0:
            corall = (cor / nac).clone()
        else:
            corall = corall + cor / nac
    u,s,vh = torch.svd(corall, compute_uv=True) 
    xac_0 = xac[:,0:num1] @ u.clone()
    x_test_0 = x_test[:,0:num1] @ u.clone()

    
    
    dataset1 = copy.deepcopy(dataset)
    dataset1['train_input'] = copy.copy(xac_0)
    dataset1['test_input']=copy.copy(x_test_0)    
    return dataset1, xac.cpu().detach().numpy(), xac_0.cpu().detach().numpy(),\
    u, corall.cpu().detach().numpy()
# In[4]:


# Generate sample data
x_train1 = torch.linspace(-1, 1, steps=100)
x_train2 = torch.linspace(-1, 1, steps=100)
x_train1_grid, x_train2_grid = torch.meshgrid(x_train1, x_train2, indexing='xy') 
x_train_all = torch.stack([x_train1_grid.reshape(-1), x_train2_grid.reshape(-1)], dim=1)

# training set
ratio = 0.1  
n_samples = int(len(x_train_all) * ratio)
random_indices = torch.randperm(len(x_train_all))[:n_samples]
x_train = x_train_all[random_indices]  
y_train = torch.tensor(target_function(x_train))

# testing set
ratio = 0.1  
n_samples = int(len(x_train_all) * ratio)
random_indices = torch.randperm(len(x_train_all))[:n_samples]
x_test0 = x_train_all[random_indices] 
y_test0 = torch.tensor(target_function(x_test0))

# Instantiate models
Jacobi_model = JacobiKAN()
mlp_model = SimpleMLP()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_Jacobi = LBFGS(Jacobi_model.parameters(), lr=0.5, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.01)



# %% iteration
# Train the models LBFGS
dataset = {
    'train_input': x_train,    
    'train_label': y_train,  
    'test_input': x_test0,     
    'test_label': y_test0     
}
epochs = 1200
epochouts = 3
train_error = np.zeros((epochouts,epochs))
test_error = np.zeros((epochouts,epochs))
for epochout in range(epochouts):
    for epoch in range(epochs):
        # --- JacobiKAN (LBFGS) ---
        def closure_Jacobi():
            optimizer_Jacobi.zero_grad()
            outputs_Jacobi = Jacobi_model(x_train)
            loss_Jacobi = criterion(outputs_Jacobi, y_train)
            loss_Jacobi.backward()
            return loss_Jacobi  
        
        optimizer_Jacobi.step(closure_Jacobi).clone().detach()  
        
        # --- MLP (Adam/SGD)  ---
        optimizer_mlp.zero_grad()
        outputs_mlp = mlp_model(x_train)
        loss_mlp = criterion(outputs_mlp, y_train)
        loss_mlp.backward()
        optimizer_mlp.step()
        
        train_error[epochout,epoch] = criterion(Jacobi_model(x_train), y_train)
        test_error[epochout,epoch] = criterion(Jacobi_model(x_test0), y_test0)
        if epoch % 5 == 0:
            with torch.no_grad():
                current_loss_Jacobi = criterion(Jacobi_model(x_train), y_train)
                current_loss_Jacobi_test = criterion(Jacobi_model(x_test0), y_test0)
                current_loss_mlp = criterion(mlp_model(x_train), y_train)
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'JacobiKAN Train Loss: {current_loss_Jacobi.item():.5f}, '
                  f'JacobiKAN Test Loss: {current_loss_Jacobi_test.item():.5f}, '
                  f'MLP Loss: {current_loss_mlp.item():.5f}')
    dataset, x0, x1, u1,corall = get_direction(Jacobi_model, dataset)
    x_train = dataset['train_input']
    y_train = dataset['train_label']
    x_test0 = dataset['test_input']
    y_test0 = dataset['test_label']   
    
    if epochout == 0:
        u_rot = u1.cpu().clone() 
        Jacobi_model0 = copy.deepcopy(Jacobi_model)  
    elif epochout <= epochouts-2:
        u_rot = u_rot @ u1.cpu().clone()     
    

# --------------- plot --------------------------
x1 = torch.linspace(-1, 1, steps=400) 
x2 = torch.linspace(-1, 1, steps=400)  
x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='xy') 
x_test = torch.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], dim=1) 

# 模型预测
y_pred_Jacobi = Jacobi_model(x_test).detach().reshape(400, 400).numpy()  # JacobiKAN
y_pred_mlp = mlp_model(x_test).detach().reshape(400, 400).numpy()           # MLP
y_true = target_function(x_test).detach().reshape(400, 400).numpy()           # true

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

# JacobiKAN prediction
im1 = ax1.contourf(x1_grid.numpy(), x2_grid.numpy(), y_pred_Jacobi, levels=20, vmin=-0.9, vmax=0.9, cmap='viridis')
ax1.set_title("JacobiKAN Predictions")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
plt.colorbar(im1, ax=ax1)

# MLP prediction
im2 = ax2.contourf(x1_grid.numpy(), x2_grid.numpy(), y_pred_mlp, levels=20, vmin=-0.9, vmax=0.9, cmap='viridis')
ax2.set_title("MLP Predictions")
ax2.set_xlabel("x1")
plt.colorbar(im2, ax=ax2)

# accurate
im3 = ax3.contourf(x1_grid.numpy(), x2_grid.numpy(), y_true, levels=20, vmin=-0.9, vmax=0.9, cmap='viridis')
ax3.set_title("accurate results")
ax3.set_xlabel("x1")
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()




