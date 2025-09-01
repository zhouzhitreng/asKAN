#!/usr/bin/env python
# coding: utf-8

# # Example 2: Possion equation
# 


# In[1]:


from kan import *
import sys
import copy
import time

device = torch.device('cpu')
print(device)

# initialize KAN with G=3
width=[2,3,1]
grid=5
k = 3
seed=1
model = KAN(width, grid, k, seed, device=device)

# create dataset
sol_fun = lambda x: torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])
source_fun = lambda x: -2*torch.pi**2 * torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])        

dim = 2
np_i = 21 # number of interior points (along each dimension)
np_b = 21 # number of boundary points (along each dimension)
ranges = [-1, 1]



sampling_mode = 'random' # 'radnom' or 'mesh'

x_mesh = torch.linspace(ranges[0],ranges[1],steps=np_i)
y_mesh = torch.linspace(ranges[0],ranges[1],steps=np_i)
X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")

# %% inner points 
x_i = torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
x_i = x_i.to(device)

# %% boundary points
# boundary, 4 sides
helper = lambda X, Y: torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
xb1 = helper(X[0], Y[0])
xb2 = helper(X[-1], Y[0])
xb3 = helper(X[:,0], Y[:,0])
xb4 = helper(X[:,0], Y[:,-1])
x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)
x_b = x_b.to(device)

# %% train
steps = 200
alpha = 0.01
log = 1

def train(x_i,x_b, u_rot, model):
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='description', ncols=100)
    losses = {
        'train_losses': [],
        'pde_losses': [],
        'bc_losses': [],
        'l2_losses': []
            }

#

    for _ in pbar:
        def closure():
            global pde_loss, bc_loss, train_loss
            optimizer.zero_grad()
            # interior loss
            sol = sol_fun(x_i)                   
                       
            x_i.requires_grad = True
            x_i1 = x_i@u_rot
            x_b1 = x_b@u_rot
            ymodel = model(x_i1)
            Dx_i = torch.autograd.grad(ymodel, x_i, grad_outputs=torch.ones_like(ymodel), create_graph=True, )
            DDxx = torch.autograd.grad(Dx_i[0][:,0], x_i, grad_outputs=torch.ones_like(Dx_i[0][:,0]), create_graph=True, )[0]
            DDXX = DDxx[:,0]
            DDyy = torch.autograd.grad(Dx_i[0][:,1], x_i, grad_outputs=torch.ones_like(Dx_i[0][:,1]), create_graph=True, )[0]
            DDYY = DDyy[:,1]
            lap = DDXX + DDYY
            lap1 = lap.view(-1,1)
            source = source_fun(x_i)
            pde_loss = torch.mean((lap1 - source)**2)

            # boundary loss
            bc_true = sol_fun(x_b)
            bc_pred = model(x_b1)
            bc_loss = torch.mean((bc_pred-bc_true)**2)

            loss = alpha * pde_loss + bc_loss
            train_loss = loss
            loss.backward()
            return loss

        x_i1 = x_i@u_rot
        if _ % 5 == 0 and _ < 50:            
            model.update_grid_from_samples(x_i1)

        optimizer.step(closure)
        sol = sol_fun(x_i)
        loss = alpha * pde_loss + bc_loss
        l2 = torch.mean((model(x_i1) - sol)**2)
        losses['train_losses'].append(loss.item())
        losses['pde_losses'].append(pde_loss.item())
        losses['bc_losses'].append(bc_loss.item())
        losses['l2_losses'].append(l2.item())

        if _ % log == 0:
            pbar.set_description("pde loss: %.2e | bc loss: %.2e | l2: %.2e " % (pde_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(), l2.cpu().detach().numpy()))
    return losses 
u0 = torch.eye(dim)

start_time = time.time()

results_model = train(x_i,x_b,u0, model)

end_time1 = time.time()
#sys.exit()
# %% active subspace

def get_direction_true(sol_fun, x_i, x_b):
    uin = torch.eye(dim)
    x_combined0 = torch.cat((x_i@uin, x_b@uin), dim=0)
    #x_combined0 = x_i@uin
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
    x_bd = x_b[:,0:num1] @ u.clone()

   
    return u, s.cpu().detach().numpy(), corall.cpu().detach().numpy()
# %% active subspace
def get_direction(model, x_i, x_b, uin):
    x_combined0 = torch.cat((x_i@uin, x_b@uin), dim=0)
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
    x_bd = x_b[:,0:num1] @ u.clone()


   
    return u, s.cpu().detach().numpy(), corall.cpu().detach().numpy()
# In[2]:



u1, s1, corall1 = get_direction(model, x_i, x_b, u0)
model1 = KAN(width, grid, k, seed, device=device)
results_model1 = train(x_i, x_b, (u0@u1).clone().detach(), model1)
model1.plot()
#sys.exit()
end_time2 = time.time()

u2, s2, corall2 = get_direction(model1, x_i, x_b, (u0@u1).clone().detach())
model2 = KAN(width, grid, k, seed, device=device)
results_model2 = train(x_i, x_b, (u0@u1@u2).clone().detach(), model2)
model2.plot()

end_time3 = time.time()

u3, s3, corall3 = get_direction(model2, x_i, x_b, (u0@u1@u2).clone().detach())
model3 = KAN(width, grid, k, seed, device=device)
results_model3 = train(x_i, x_b, (u0@u1@u2@u3).clone().detach(), model3)
model3.plot()

end_time4 = time.time()

u4, s4, corall4 = get_direction(model3, x_i, x_b, (u0@u1@u2@u3).clone().detach())
model4 = KAN(width, grid, k, seed, device=device)
results_model4  = train(x_i, x_b, (u0@u1@u2@u3@u4).clone().detach(), model4)
model4.plot()

end_time5 = time.time()

u5, s5, corall5 = get_direction(model4, x_i, x_b, (u0@u1@u2@u3@u4).clone().detach())
model5 = KAN(width, grid, k, seed, device=device)
results_model5 = train(x_i, x_b, (u0@u1@u2@u3@u4@u5).clone().detach(), model5)
model5.plot()



# %%  prepare data for plot

x_min, x_max = -1, 1
y_min, y_max = -1, 1
x_steps, y_steps = 100, 100


x_coords = torch.linspace(x_min, x_max, x_steps)
y_coords = torch.linspace(y_min, y_max, y_steps)


x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

xgrid = torch.stack((x_grid, y_grid), dim=-1)

x0 = xgrid.view(-1, 2)  

# accurate 
y0 = sol_fun(x0)

# initial model
y1 = model.cpu()(x0.cpu())


# 
x1 = x0.cpu() @ u1.cpu().clone() 
x2 = x1.cpu() @ u2.cpu().clone() 
x3 = x2.cpu() @ u3.cpu().clone() 
x4 = x3.cpu() @ u4.cpu().clone() 
x5 = x4.cpu() @ u5.cpu().clone() 

# 
uu1 = u1.cpu().clone()
uu2 = u1.cpu().clone()@u2.cpu().clone()
uu3 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()
uu4 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()
uu5 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()@u5.cpu().clone()

#
y2 = model1.cpu()(x1.cpu())


# %% plot

#
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

#
X, Y = np.meshgrid(x, y)

# 
y0_np = y0.detach().numpy().reshape(100, 100)
y1_np = y1.detach().numpy().reshape(100, 100)
y2_np = y2.detach().numpy().reshape(100, 100)

# 
vmin = min(y0_np.min(), y1_np.min(), y2_np.min())
vmax = max(y0_np.max(), y1_np.max(), y2_np.max())

# 
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# y0
im0 = axs[0].pcolormesh(X, Y, y0_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('True Solution y0')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# y1
im1 = axs[1].pcolormesh(X, Y, y1_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model Solution y1')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

# y2
im2 = axs[2].pcolormesh(X, Y, y2_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[2].set_title('Transformed Model Solution y2')
axs[2].set_xlabel('x0_1')
axs[2].set_ylabel('x0_2')


# 
plt.tight_layout()

plt.show()



