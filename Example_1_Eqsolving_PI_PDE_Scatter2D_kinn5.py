#!/usr/bin/env python
# coding: utf-8

# # Example 1: Function Fitting
# 
# In this example, we will cover how to leverage grid refinement to maximimze KANs' ability to fit functions

# intialize model and create dataset

# In[1]:

# 测试一下远场噪声重构
# 注意：
#       输入为幅值a与相位cta，
#       噪声为a*cos(wt+cta)

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
#device = torch.device('cpu')
print(device)


# 两次训练的训练时间
time_1st = 2046.056447505951
time_2nd = 2065.1926305294037

# initialize KAN with G=3
width=[3,5,1]
grid=5
k = 3
seed=1

model = KAN(width, grid, k, seed, device=device)

# create dataset
#f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
# f = lambda x: (torch.cos(12*x[:,[0]]**2 + 12*x[:,[1]]**2))
# f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]])))+\
#               0.2*x[:,[0]]+0.1*x[:,[1]])
# f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]])))+\
#               0.7*x[:,[0]]+x[:,[1]])
# %% physical parameter
c = 1
ac_lam = 2*np.pi
omega = 1
dim = 3
# 获取局部声场数据
# 加载 .mat 文件
mat_data = scipy.io.loadmat('C_train_2D.mat')

# 访问文件中的变量
C_train = mat_data['C_train']
scale = 10**5

# 获取数据的维度
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


# %% 绘制图片，dataset0某一时刻的空间压力分布
plt.plot
# 从dataset0中随机选取500个点
rand_indices = torch.randperm(dataset0.size(0))[:1000].to(device)

# 生成训练测试集前四列的数据
x_i = dataset0[:,0:3]
# 生成训练测试集最后一列的数据
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
# x_b为旋转后的边界点
# x_i为旋转后的内点，与边界点一起，都是需要不断更新
# x_b0为原始边界点
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.01, nesterov=True)
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
            
            # rand_indices = torch.randperm(sol.size(0))[:500] # 默认500       
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
            
            # 计算 Hessian 矩阵
            # H = hessian(lambda x: model(x).sum(), x_i)
            
            # # 提取对角线元素作为二阶导数
            # DDXX = H[:, 0, 0]
            # DDYY = H[:, 1, 1]
            # DDZZ = H[:, 2, 2]
            # DDTT = H[:, 3, 3]
            
            lap = DDTT - DDXX - DDYY
            lap1 = lap.view(-1,1)
            pde_loss = torch.mean((lap1.squeeze())**2)
            
            value_loss = torch.mean((model(x_i1[rand_indices]).squeeze()-selected_rows)**2)        
            #pde_loss = value_loss - value_loss
            

            
            #loss = alpha * pde_loss + bc_loss + bc_loss1
            loss = alpha*pde_loss + value_loss
            #loss = value_loss
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
# 从模型中获得活跃子空间
def get_direction(model, x_i, uin):
    x_i = x_i.to(device)
    model = model.to(device)
    uin = uin.to(device)
    #x_combined0 = torch.cat((x_i@uin), dim=0)
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
    # xac_0[:, 0] *= s[0]
    # xac_0[:, 1] *= s[1]
    # x_test_0[:, 0] *= s[0]

   
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

# u4, s4, corall4 = get_direction(model3, x_i, x_b, (u0@u1@u2@u3).clone().detach())
# model4 = KAN(width, grid, k, seed, device=device)
# results_model4 = train(x_i, x_b, x_b1, (u0@u1@u2@u3@u4).clone().detach(), model4)
# model4.plot()

# u5, s5, corall5 = get_direction(model4, x_i, x_b, (u0@u1@u2@u3@u4).clone().detach())
# model5 = KAN(width, grid, k, seed, device=device)
# results_model5 = train(x_i, x_b, x_b1, (u0@u1@u2@u3@u4@u5).clone().detach(), model5)
# model5.plot()



# dataset6, x0, x1, u6,corall = get_direction(model, dataset5)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset6, opt="LBFGS", steps=300);

# dataset7, x0, x1, u7,corall = get_direction(model, dataset6)
# m1odel = KAN(width, grid, k, seed, device=device)
# model.fit(dataset7, opt="LBFGS", steps=300);

# dataset8, x0, x1, u8,corall = get_direction(model, dataset7)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset8, opt="LBFGS", steps=300);

#sys.exit()

## 绘制图片
for ii in range(nx):
    for jj in range(ny):
        for tt in range(t_steps):
                order = tt+jj*t_steps+ii*t_steps*ny
                dataset0[order-1,0] = C_train[ii,jj,0]-100
                dataset0[order-1,1] = C_train[ii,jj,1]-100
                dataset0[order-1,2] = t[tt]-5
                dataset0[order-1,3] = scale*C_train[ii,jj,2]*torch.cos(omega*t[tt]+C_train[ii,jj,3])
                
                

# %% 精确解，不同时刻
import matplotlib.tri as tri
# 加载 .mat 文件
mat_data = scipy.io.loadmat('C_train_2D_plot_1.mat')
# 访问文件中的变量
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
    
    # 绘制等高线填充图
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50, vmin=-8, vmax=8)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  # 设置 extendfrac=0
    cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置颜色条的刻度范围
    cbar.set_ticks(np.linspace(-8, 8, 9))  # 设置刻度位置
    cbar.set_ticklabels(np.linspace(-8, 8, 9))  # 设置刻度标签
    
    # 添加颜色条
    #cbar = fig.colorbar(contour, ax=axs)
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置标题和坐标轴标签
    # axs.set_title('Contour Plot of Model Output', fontsize=35)
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    
    # 设置坐标轴范围
    plt.ylim(-12, 12)  # 确保纵轴范围为 0 到 0.5
    plt.xlim(-12, 12)    # 确保横轴范围为 0 到 1
    
    # 设置坐标轴刻度
    plt.yticks(np.arange(-12, 12.1, 6))
    plt.xticks(np.arange(-12, 12.1, 6))
    
    # 设置小刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    
    # 设置刻度方向和粗细
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=5, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


# %% KAN模型，不同时刻
import matplotlib.tri as tri
# 加载 .mat 文件
mat_data = scipy.io.loadmat('C_train_2D_plot.mat')
# 访问文件中的变量
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
            #XYTIN = np.array([X[ip],Y[ip],t[tt*ntgap]-5])
            input_tensor[ip] = torch.tensor([X[ipp], Y[ipp], t[tt*ntgap]-5], dtype=torch.float32)
            ip += 1
        ipp +=1
# 将输入张量传递给模型
model = model.cpu()  # 确保模型在 CPU 上
y00 = model(input_tensor)
# 将输出转换为 NumPy 数组并重塑为 (nx, ny, nnt)
Z0 = y00.detach().numpy().reshape(nx, ny, nnt) * scale_plot / scale
# 按照 X, Y 的排序重新排列 Z0
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
    
    # 绘制等高线填充图
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50, vmin=-8, vmax=8)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    contour = axs.tricontourf(triang, Z_kan[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  # 设置 extendfrac=0
    cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置颜色条的刻度范围
    cbar.set_ticks(np.linspace(-8, 8, 9))  # 设置刻度位置
    cbar.set_ticklabels(np.linspace(-8, 8, 9))  # 设置刻度标签
    
    # 添加颜色条
    #cbar = fig.colorbar(contour, ax=axs)
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置标题和坐标轴标签
    # axs.set_title('Contour Plot of Model Output', fontsize=35)
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    
    # 设置坐标轴范围
    plt.ylim(97, 103)  # 确保纵轴范围为 0 到 0.5
    plt.xlim(97, 103)    # 确保横轴范围为 0 到 1
    
    # 设置坐标轴刻度
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    
    # 设置小刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    
    # 设置刻度方向和粗细
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    #plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


# %% ASM-KAN模型，不同时刻
import matplotlib.tri as tri
# 加载 .mat 文件
mat_data = scipy.io.loadmat('C_train_2D_plot_1.mat')
# 访问文件中的变量
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
            #XYTIN = np.array([X[ip],Y[ip],t[tt*ntgap]-5])
            input_tensor[ip] = torch.tensor([X[ipp], Y[ipp], t[tt*ntgap]-5], dtype=torch.float32)
            ip += 1
        ipp +=1
# 将输入张量传递给模型
model1 = model1.cpu()  # 确保模型在 CPU 上
input_tensor1 = input_tensor @ u1.cpu()
y00 = model1(input_tensor1)
# 将输出转换为 NumPy 数组并重塑为 (nx, ny, nnt)
Z0 = y00.detach().numpy().reshape(nx, ny, nnt) * scale_plot / scale
# 按照 X, Y 的排序重新排列 Z0
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
    
    # 绘制等高线填充图
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50, vmin=-8, vmax=8)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    contour = axs.tricontourf(triang, Z_asmkan[:,ipic], cmap='YlOrBr', levels=100, vmin=-8, vmax=8, extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=axs, extend='both', extendfrac=0)  # 设置 extendfrac=0
    cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置颜色条的刻度范围
    cbar.set_ticks(np.linspace(-8, 8, 9))  # 设置刻度位置
    cbar.set_ticklabels(np.linspace(-8, 8, 9))  # 设置刻度标签
    
    # 添加颜色条
    #cbar = fig.colorbar(contour, ax=axs)
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置标题和坐标轴标签
    # axs.set_title('Contour Plot of Model Output', fontsize=35)
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    
    # 设置坐标轴范围
    plt.ylim(97, 103)  # 确保纵轴范围为 0 到 0.5
    plt.xlim(97, 103)    # 确保横轴范围为 0 到 1
    
    # 设置坐标轴刻度
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    
    # 设置小刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    
    # 设置刻度方向和粗细
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    #plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


# %% 误差，不同时刻
import matplotlib.tri as tri
from matplotlib.colors import BoundaryNorm
# 加载 .mat 文件
mat_data = scipy.io.loadmat('C_train_2D_plot.mat')
# 访问文件中的变量
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
error_kan = np.zeros((nx*ny,nnt))
error_asmkan = np.zeros((nx*ny,nnt))
input_tensor = torch.zeros(nx * ny * nnt, 3)
for ii in range(nx):
    for jj in range(ny):        
        for tt in range(0,nnt):
            X[ipp] = C_train_plot[ii,jj,0] - 100
            Y[ipp] = C_train_plot[ii,jj,1] - 100
        ipp +=1
# 将输入张量传递给模型
model = model.cpu()  # 确保模型在 CPU 上
y00 = model(input_tensor)
# 将输出转换为 NumPy 数组并重塑为 (nx, ny, nnt)
Z0 = y00.detach().numpy().reshape(nx, ny, nnt) * scale_plot / scale
# 按照 X, Y 的排序重新排列 Z0
ip = 0
for ii in range(nx):
    for jj in range(ny):
        for tt in range(0,nnt):
            error_kan[ip,tt] = abs(Z_kan[ip,tt]-Z[ip,tt])
            error_asmkan[ip,tt] = abs(Z_asmkan[ip,tt]-Z[ip,tt])
        ip = ip + 1

# kan误差
for ipic in range(0,nnt):
    fig_name = "result/error_kan_"+str(ipic)+".png"
    X1 = X + 100
    Y1 = Y + 100   
    triang = tri.Triangulation(X1,Y1)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    
    # 绘制等高线填充图
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50, vmin=-8, vmax=8)
    
    # 设置颜色条的范围
    vmin, vmax = 0, 0.5
    levels = np.linspace(vmin, vmax, 100)  # 生成 100 个等高线级别
    norm = BoundaryNorm(levels, ncolors=256)
    
    # 绘制等高线图
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    contour = axs.tricontourf(triang, error_kan[:, ipic], cmap='YlOrBr', levels=levels, norm=norm)
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=axs)
    cbar.ax.tick_params(labelsize=20)
    
    # 设置颜色条的刻度范围
    ticks = np.linspace(vmin, vmax, 6)  # 设置刻度位置
    tick_labels = np.round(ticks, 2)  # 格式化刻度标签为两位小数
    cbar.set_ticks(ticks)  # 设置刻度位置
    cbar.set_ticklabels(tick_labels)  # 设置刻度标签
    
    # 确保只显示指定的刻度标签
    cbar.ax.yaxis.set_ticks(ticks)
    cbar.ax.yaxis.set_ticklabels(tick_labels)
    
    # 移除小刻度
    cbar.ax.yaxis.set_tick_params(which='minor', length=0)  # 移除小刻度
    # 移除默认的刻度标签
    #cbar.ax.yaxis.set_ticks_position('none')  # 移除默认的刻度标签
    # 添加颜色条
    #cbar = fig.colorbar(contour, ax=axs)
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置标题和坐标轴标签
    # axs.set_title('Contour Plot of Model Output', fontsize=35)
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    
    # 设置坐标轴范围
    plt.ylim(97, 103)  # 确保纵轴范围为 0 到 0.5
    plt.xlim(97, 103)    # 确保横轴范围为 0 到 1
    
    # 设置坐标轴刻度
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    
    # 设置小刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    
    # 设置刻度方向和粗细
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()
    
# kan误差
for ipic in range(0,nnt):
    fig_name = "result/error_asmkan_"+str(ipic)+".png"
    X1 = X + 100
    Y1 = Y + 100   
    triang = tri.Triangulation(X1,Y1)
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    
    # 绘制等高线填充图
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50)
    
    # 绘制等高线填充图，固定颜色范围在 -8 到 8
    # contour = axs.tricontourf(triang, Z[:,ipic], cmap='YlOrBr', levels=50, vmin=-8, vmax=8)
    
    # 设置颜色条的范围
    vmin, vmax = 0, 0.5
    levels = np.linspace(vmin, vmax, 100)  # 生成 100 个等高线级别
    norm = BoundaryNorm(levels, ncolors=256)
    
    # 绘制等高线图
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    contour = axs.tricontourf(triang, error_asmkan[:, ipic], cmap='YlOrBr', levels=levels, norm=norm)
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=axs)
    cbar.ax.tick_params(labelsize=20)
    
    # 设置颜色条的刻度范围
    ticks = np.linspace(vmin, vmax, 6)  # 设置刻度位置
    tick_labels = np.round(ticks, 2)  # 格式化刻度标签为两位小数
    cbar.set_ticks(ticks)  # 设置刻度位置
    cbar.set_ticklabels(tick_labels)  # 设置刻度标签
    
    # 确保只显示指定的刻度标签
    cbar.ax.yaxis.set_ticks(ticks)
    cbar.ax.yaxis.set_ticklabels(tick_labels)
    
    # 移除小刻度
    cbar.ax.yaxis.set_tick_params(which='minor', length=0)  # 移除小刻度
    # 移除默认的刻度标签
    #cbar.ax.yaxis.set_ticks_position('none')  # 移除默认的刻度标签
    # 添加颜色条
    #cbar = fig.colorbar(contour, ax=axs)
    #cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('p', fontsize=25)  # 设置颜色条标签
    
    # 设置标题和坐标轴标签
    # axs.set_title('Contour Plot of Model Output', fontsize=35)
    axs.set_xlabel(r'$x_1~/~\lambda$', fontsize=25)
    axs.set_ylabel(r'$x_2~/~\lambda$', fontsize=25)
    
    # 设置坐标轴范围
    plt.ylim(97, 103)  # 确保纵轴范围为 0 到 0.5
    plt.xlim(97, 103)    # 确保横轴范围为 0 到 1
    
    # 设置坐标轴刻度
    plt.yticks(np.arange(97, 103.1, 1))
    plt.xticks(np.arange(97, 103.1, 1))
    
    # 设置小刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    
    # 设置刻度方向和粗细
    plt.tick_params(axis='y', direction='in')
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')
    plt.tick_params(axis='both', labelsize=20)
    
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()

# %% plt the error
# 假设 results_model1 是一个字典，包含训练和测试损失
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator, AutoMinorLocator
rc('font',**{'family':'serif','serif':['Times New Roman']})
train_loss1 = np.array(results_model['l2_losses'])
test_loss1 = np.array(results_model1['l2_losses'])
figname = 'result/Sound_reconstruction_Loss.png'

# 创建图形
plt.figure(figsize=(12, 5))

# 绘制训练损失，使用实线，颜色为较深的蓝色，线条粗细为2
plt.plot(train_loss1/scale**2, label='KAN', linestyle='-', marker = 'o', color='blue', markevery=170, markersize=10, linewidth=2)

# 绘制测试损失，使用虚线，颜色为较浅的蓝色，线条粗细为2
plt.plot(test_loss1/scale**2, label='ASM-KAN', linestyle='--', marker = 'v', color='DodgerBlue', markevery=200, markersize=10, linewidth=2)

# 添加图例，并设置字体大小
#plt.legend(fontsize=25)
plt.legend(fontsize=25, framealpha=0.5, facecolor='gray')

# 添加轴标签，并设置字体大小
# 设置横轴范围
plt.xlim(-10, 500)  # 确保横轴范围为 0.1 到 1


# 设置纵轴范围
#plt.ylim(-0.005, 10)  # 确保横轴范围为 0.1 到 1

# 设置y轴为对数坐标
plt.yscale('log')
plt.ylim(10**0/scale**2, 10**4/scale**2) # 需要根据不同的算例进行调整
#plt.yticks([10**-5,10**-4,10**-3,10**-2,10**-1], [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],fontsize=30)

# 设置纵轴的刻度为每隔 0.1 一个
#plt.yticks(np.arange(0.0, 10, 0.02))

# 添加小刻度
# 设置小刻度，每两个大刻度之间一个小刻度
#plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
y_ticks = [10**i for i in range(-10, -5)]  # 从 10^-5 到 10^0
plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
# 设置y轴刻度样式，确保右边也有小刻度
plt.tick_params(axis='y', direction='in', right=True, labelright=False, which='both')

# 设置横轴的刻度为每隔 50 一个
plt.xticks(np.arange(0, 501, 100))

# 设置字体为 Times New Roman
plt.rc('font', family='Times New Roman')

# 设置刻度向内
# plt.tick_params(axis='y', direction='in', right=True, labelright=False)
# plt.tick_params(axis='y', direction='in', right=True, labelright=False)

# 设置刻度的粗细
plt.tick_params(axis='both', which='major', width=2, length=8)  # 设置大刻度的粗细
plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')  # 设置小刻度的粗细并向内

# 确保不显示小刻度的值
plt.gca().tick_params(axis='y', which='minor', labelleft=False)    # 不显示小刻度的值
plt.gca().tick_params(axis='x', which='minor', labelbottom=False)  # 不显示小刻度的值

# 添加图例（无背景），设置为两列，并调整线条与文本之间的距离
#plt.legend(frameon=False, fontsize=25, loc='upper left', bbox_to_anchor=(0.0, 0.35), ncol=2, handletextpad=0.5)  # 向右上角移动图例

# 设置坐标轴标签

plt.xlabel('Epochs', fontsize=35)  # X 轴标签为 LaTeX 形式的 theta
plt.ylabel('Loss', fontsize=35)  # Y 轴标题为 |p'|

# 设置坐标轴刻度标签的字体大小
plt.tick_params(axis='both', labelsize=30)  # 将坐标轴刻度标签的字体大小设置为30磅

# 保存图形为 600 dpi
plt.savefig(figname, dpi=600, bbox_inches='tight')

# 显示图表
plt.show()









