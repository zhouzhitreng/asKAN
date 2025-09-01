#!/usr/bin/env python
# coding: utf-8

# # Example 3： acoustic reconstruction 

# In[1]:


from kan import *
import sys
import copy
from torch.autograd.functional import hessian
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import scipy
import sys
import numpy as np
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# initialize KAN with G=3
width=[3,5,1]
grid=5
k = 3
seed=1

model = KAN(width, grid, k, seed, device=device)


# %% physical parameter
c = 1
ac_lam = 2*np.pi
omega = 1
dim = 3

# load data
mat_data = scipy.io.loadmat('C_train_2D.mat')

C_train = mat_data['C_train']
scale = 10**5


t_min = 0; t_max = 10; t_steps = 200;
t = torch.linspace(t_min,t_max,t_steps)
nx = len(C_train[:,1,1])
ny = len(C_train[1,:,1])
dataset0 = torch.zeros(nx*ny*t_steps,5)

for ii in range(nx):
    for jj in range(ny):
        for tt in range(t_steps):
                order = tt+jj*t_steps+ii*t_steps*ny
                dataset0[order-1,0] = C_train[ii,jj,0]-100
                dataset0[order-1,1] = C_train[ii,jj,1]-100
                dataset0[order-1,2] = t[tt]-5
                dataset0[order-1,3] = scale*C_train[ii,jj,2]*torch.cos(omega*t[tt]+C_train[ii,jj,3])


# %% 
plt.plot
# 
rand_indices = torch.randperm(dataset0.size(0))[:1000].to(device)


x_i = dataset0[:,0:3]
sol = dataset0[:,3]
steps = 500
alpha = 1
alpha1 = 1
log  = 1

def train(x_i, sol, u_rot, model):
    x_i = x_i.to(device)
    sol = sol.to(device)
    model = model.to(device)
    u_rot = u_rot.to(device)
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    pbar = tqdm(range(steps), desc='description', ncols=100)
    losses = {
        'train_losses': [],
        'pde_losses': [],
        'l2_losses': []
            }
    for _ in pbar:
        def closure():
            global pde_loss, value_loss
            optimizer.zero_grad()
            # interior loss           
                       
            x_i.requires_grad = True                
            selected_rows = sol[rand_indices]            
            x_i1 = x_i@u_rot
            
            ymodel = model(x_i1)
            Dx_i = torch.autograd.grad(ymodel, x_i, grad_outputs=torch.ones_like(ymodel), create_graph=True, )
            DDxx = torch.autograd.grad(Dx_i[0][:,0], x_i, grad_outputs=torch.ones_like(Dx_i[0][:,0]), create_graph=True, )[0]
            DDXX = DDxx[:,0]
            DDyy = torch.autograd.grad(Dx_i[0][:,1], x_i, grad_outputs=torch.ones_like(Dx_i[0][:,1]), create_graph=True, )[0]
            DDYY = DDyy[:,1]
            DDtt = torch.autograd.grad(Dx_i[0][:,2], x_i, grad_outputs=torch.ones_like(Dx_i[0][:,2]), create_graph=True, )[0]
            DDTT = DDtt[:,2]
            lap = DDTT - DDXX - DDYY
            lap1 = lap.view(-1,1)
            pde_loss = torch.mean((lap1.squeeze())**2)
            
            value_loss = torch.mean((model(x_i1[rand_indices]).squeeze()-selected_rows)**2)        


            loss = alpha*pde_loss + value_loss
            loss.backward()
            return loss

        x_i1 = x_i@u_rot
        if _ % 5 == 0 and _ < 50:            
            model.update_grid_from_samples(x_i1)

        optimizer.step(closure)
        loss = alpha * pde_loss  + alpha1 * value_loss
        l2 = torch.mean((model(x_i1).squeeze() - sol)**2)
        losses['train_losses'].append(loss.item())
        losses['pde_losses'].append(pde_loss.item())
        losses['l2_losses'].append(l2.item())

        if _ % log == 0:
            pbar.set_description("pde loss: %.2e | l2: %.2e| value_loss: %.2e " % (pde_loss.cpu().detach().numpy(),                                 
                                                                                   l2.cpu().detach().numpy(),value_loss))
    return losses
u0 = torch.eye(dim)

start_time = time.time()
results_model= train(x_i,sol,u0, model)
#sys.exit()
end_time1 = time.time()
# %% active subspace

def get_direction(model, x_i, uin):
    x_i = x_i.to(device)
    model = model.to(device)
    uin = uin.to(device)
    x_combined0 = torch.cat((x_i@uin,), dim=0)
    x_combined  = x_combined0.clone().detach()
    x_combined.requires_grad = True
    ymodel = model(x_combined)
    deri_x_i = torch.autograd.grad(ymodel, x_combined, grad_outputs=torch.ones_like(ymodel), create_graph=True, )
    num1 = len(x_combined[0,:])
    nac  = len(x_combined[:,0])
    corall = torch.zeros((num1, num1))
    for i1 in range(nac):
        arr1_col = deri_x_i[0][i1,:]
        vector_col = arr1_col.unsqueeze(1)
        arr1_row = arr1_col.T    
        cor = vector_col @ vector_col.T
        if i1 == 0:
            corall = (cor / nac).clone()
        else:
            corall = corall + cor / nac
    u,s,vh = torch.svd(corall, compute_uv=True) 
    x_in = x_i[:,0:num1] @ u.clone()

   
    return u, s.cpu().detach().numpy(), corall.cpu().detach().numpy()
# In[2]:


u1, s1, corall1 = get_direction(model, x_i, u0)
model1 = KAN(width, grid, k, seed, device=device)
results_model1 = train(x_i, sol, (u0.to(device)@u1.to(device)).clone().detach(), model1)
model1.plot()


end_time2 = time.time()
sys.exit()

u2, s2, corall2 = get_direction(model1, x_i, (u0.to(device)@u1.to(device)).clone().detach())
model2 = KAN(width, grid, k, seed, device=device)
results_model2 = train(x_i, sol, (u0.to(device)@u1.to(device)@u2.to(device)).clone().detach(), model2)
model2.plot()

u3, s3, corall3 = get_direction(model2, x_i, (u0.to(device)@u1.to(device)@u2.to(device)).clone().detach())
model3 = KAN(width, grid, k, seed, device=device)
results_model3 = train(x_i, sol, (u0.to(device)@u1.to(device)@u2.to(device)@u3.to(device)).clone().detach(), model3)
model3.plot()


## prepare data for plot
for ii in range(nx):
    for jj in range(ny):
        for tt in range(t_steps):
                order = tt+jj*t_steps+ii*t_steps*ny
                dataset0[order-1,0] = C_train[ii,jj,0]-100
                dataset0[order-1,1] = C_train[ii,jj,1]-100
                dataset0[order-1,2] = t[tt]-5
                dataset0[order-1,3] = scale*C_train[ii,jj,2]*torch.cos(omega*t[tt]+C_train[ii,jj,3])
                
                

# %%  accurate value
import matplotlib.tri as tri
mat_data = scipy.io.loadmat('C_train_2D_plot_1.mat')
C_train_plot = mat_data['C_train_plot']
ip = 0
ntgap = 20
nx = 200
ny = 200
nnt = 4
scale_plot = 10**3
X = np.zeros(nx*ny)
Y = np.zeros(nx*ny)
Z = np.zeros((nx*ny,nnt))
for ii in range(nx):
    for jj in range(ny):        
        for tt in range(0,nnt):
            X[ip] = C_train_plot[ii,jj,0]
            Y[ip] = C_train_plot[ii,jj,1]
            Z[ip,tt] = scale_plot*C_train_plot[ii,jj,2]*torch.cos(omega*t[tt*ntgap]+C_train_plot[ii,jj,3])
        ip = ip + 1

for ipic in range(0,nnt):
    fig_name = "result/new_acc_"+str(ipic)+".png"           
    triang = tri.Triangulation(X,Y)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  # 设置 extendfrac=0
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.linspace(-8, 8, 9))  # 设置刻度位置
    cbar.set_ticklabels(np.linspace(-8, 8, 9))  # 设置刻度标签
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    plt.ylim(-12, 12)  
    plt.xlim(-12, 12)  
    plt.yticks(np.arange(-12, 12.1, 6))
    plt.xticks(np.arange(-12, 12.1, 6))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=5, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


# %% KAN
import matplotlib.tri as tri
mat_data = scipy.io.loadmat('C_train_2D_plot.mat')
C_train_plot = mat_data['C_train_plot']
ip = 0
ipp = 0
ntgap = 20
nx = 100
ny = 100
nnt = 4
scale_plot = 10**4
X = np.zeros(nx*ny)
Y = np.zeros(nx*ny)
Z_kan = np.zeros((nx*ny,nnt))
input_tensor = torch.zeros(nx * ny * nnt, 3)
for ii in range(nx):
    for jj in range(ny):        
        for tt in range(0,nnt):
            X[ipp] = C_train_plot[ii,jj,0] - 100
            Y[ipp] = C_train_plot[ii,jj,1] - 100
            input_tensor[ip] = torch.tensor([X[ipp], Y[ipp], t[tt*ntgap]-5], dtype=torch.float32)
            ip += 1
        ipp +=1
model = model.cpu() 
y00 = model(input_tensor)
Z0 = y00.detach().numpy().reshape(nx, ny, nnt) * scale_plot / scale
ip = 0
for ii in range(nx):
    for jj in range(ny):
        for tt in range(0,nnt):
            Z_kan[ip,tt] = Z0[ii,jj,tt]
        ip = ip + 1


for ipic in range(0,nnt):
    fig_name = "result/kan_"+str(ipic)+".png"
    X1 = X + 100
    Y1 = Y + 100   
    triang = tri.Triangulation(X1,Y1)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    contour = axs.tricontourf(triang, Z_kan[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  # 设置 extendfrac=0
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.linspace(-8, 8, 9))  
    cbar.set_ticklabels(np.linspace(-8, 8, 9))    
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    plt.ylim(97, 103)
    plt.xlim(97, 103)
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    plt.show()


# %% asKAN
import matplotlib.tri as tri
mat_data = scipy.io.loadmat('C_train_2D_plot_1.mat')
C_train_plot = mat_data['C_train_plot']
ip = 0
ipp = 0
ntgap = 20
nx = 100
ny = 100
nnt = 4
scale_plot = 10**4
X = np.zeros(nx*ny)
Y = np.zeros(nx*ny)
Z_asmkan = np.zeros((nx*ny,nnt))
input_tensor = torch.zeros(nx * ny * nnt, 3)
for ii in range(nx):
    for jj in range(ny):        
        for tt in range(0,nnt):
            X[ipp] = C_train_plot[ii,jj,0] - 100
            Y[ipp] = C_train_plot[ii,jj,1] - 100
            input_tensor[ip] = torch.tensor([X[ipp], Y[ipp], t[tt*ntgap]-5], dtype=torch.float32)
            ip += 1
        ipp +=1
model1 = model1.cpu()
input_tensor1 = input_tensor @ u1.cpu()
y00 = model1(input_tensor1)
Z0 = y00.detach().numpy().reshape(nx, ny, nnt) * scale_plot / scale
ip = 0
for ii in range(nx):
    for jj in range(ny):
        for tt in range(0,nnt):
            Z_asmkan[ip,tt] = Z0[ii,jj,tt]
        ip = ip + 1


for ipic in range(0,nnt):
    fig_name = "result/asm-kan_"+str(ipic)+".png"
    X1 = X + 100
    Y1 = Y + 100   
    triang = tri.Triangulation(X1,Y1)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    
    contour = axs.tricontourf(triang, Z_asmkan[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.linspace(-8, 8, 9)) 
    cbar.set_ticklabels(np.linspace(-8, 8, 9))   
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    plt.ylim(97, 103)  
    plt.xlim(97, 103)  
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    plt.show()











