#!/usr/bin/env python
# coding: utf-8


# In[1]:

# cos形式的高频行波解是一类比较好的验证算例
#  这样做可以分解出行波中的两部分
# 计算标度率，与传统的方法比较

from kan import *
import sys
import copy
import time
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


    
# create dataset


# function 1
f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]]))))
width=[2,5,1]
grid=5
k = 3
seed=1  
model = KAN(width, grid, k, seed, device=device)   


dataset = create_dataset(f, n_var=2, device=device)
start_time = time.time()

results_model = loss = model.fit(dataset, opt="LBFGS", steps=200);
model.plot()

end_time1 = time.time()
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
# In[2]:



dataset1, x0, x1, u1,corall = get_direction(model, dataset)
model1 = KAN(width, grid, k, seed, device=device)
results_model1 = model1.fit(dataset1, opt="LBFGS", steps=200);
model1.plot()

#sys.exit()
end_time2 = time.time()

dataset2, x00, x11, u2, corall = get_direction(model1, dataset1)
model2 = KAN(width, grid, k, seed, device=device)
results_model2 = model2.fit(dataset2, opt="LBFGS", steps=200);
model2.plot()

end_time3 = time.time()




# %% prepare data for ploting

x_min, x_max = -1, 1
y_min, y_max = -1, 1
x_steps, y_steps = 100, 100

x_coords = torch.linspace(x_min, x_max, x_steps)
y_coords = torch.linspace(y_min, y_max, y_steps)


x_grid, y_grid = torch.meshgrid(x_coords, y_coords)
xgrid = torch.stack((x_grid, y_grid), dim=-1)

x0 = xgrid.view(-1, 2)

# accurate
y0 = f(x0)

# initial model
y1 = model.cpu()(x0.cpu())


# independent variables
x1 = x0.cpu() @ u1.cpu().clone() 
x2 = x1.cpu() @ u2.cpu().clone() 
x3 = x2.cpu() @ u3.cpu().clone() 
x4 = x3.cpu() @ u4.cpu().clone() 
x5 = x4.cpu() @ u5.cpu().clone() 

# dependent variables
uu1 = u1.cpu().clone()
uu2 = u1.cpu().clone()@u2.cpu().clone()
uu3 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()
uu4 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()
uu5 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()@u5.cpu().clone()

# askan model
y2 = model2.cpu()(x2.cpu())


# %% PLT 

# boundary of variables
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)


X, Y = np.meshgrid(x, y)


y0_np = y0.detach().numpy().reshape(100, 100)
y1_np = y1.detach().numpy().reshape(100, 100)
y2_np = y2.detach().numpy().reshape(100, 100)


vmin = min(y0_np.min(), y1_np.min(), y2_np.min())
vmax = max(y0_np.max(), y1_np.max(), y2_np.max())

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# y0
im0 = axs[0].pcolormesh(X, Y, y0_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('True Solution y0')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# y1
im1 = axs[1].pcolormesh(X, Y, y1_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model Solution y1')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

# y2
im2 = axs[2].pcolormesh(X, Y, y2_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[2].set_title('Transformed Model Solution y2')
axs[2].set_xlabel('x0_1')
axs[2].set_ylabel('x0_2')



plt.tight_layout()


plt.show()




