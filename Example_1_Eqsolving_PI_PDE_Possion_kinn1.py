#!/usr/bin/env python
# coding: utf-8

# # Example 1: Function Fitting
# 
# In this example, we will cover how to leverage grid refinement to maximimze KANs' ability to fit functions

# intialize model and create dataset

# In[1]:

# 测试一下泊松方程
# 我这里需要区分旋转后的内点，边界点以及边界上的准确值
# 注意：这里的导数需要大于
# 注意，当矩阵接近单位阵时，矩阵的svd分解的特征向量会非常敏感

from kan import *
import sys
import copy
import time
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

# initialize KAN with G=3
width=[2,3,1]
grid=5
k = 3
seed=1
# 参数总个数为220

# 前两次迭代所花的时间
time_1st = 239.90955686569214
time_2nd = 362.76224184036255


model = KAN(width, grid, k, seed, device=device)

# create dataset
#f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
# f = lambda x: (torch.cos(12*x[:,[0]]**2 + 12*x[:,[1]]**2))
# f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]])))+\
#               0.2*x[:,[0]]+0.1*x[:,[1]])
# f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]])))+\
#               0.7*x[:,[0]]+x[:,[1]])
sol_fun = lambda x: torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])
source_fun = lambda x: -2*torch.pi**2 * torch.sin(torch.pi*x[:,[0]])*torch.sin(torch.pi*x[:,[1]])        
#torch.cos(2*torch.pi*(((0.5*x[:,[0]])+1.3*x[:,[1]])))
#f = lambda x: (torch.sin(3*torch.pi*((0.3*(x[:,[0]]+x[:,[1]])**2+0.7*x[:,[1]]**1))))
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

sys.exit()

# %% train
steps = 200
alpha = 0.01
log = 1

def train(x_i,x_b, u_rot, model):
# x_b为旋转后的边界点
# x_i为旋转后的内点，与边界点一起，都是需要不断更新
# x_b0为原始边界点
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='description', ncols=100)
    losses = {
        'train_losses': [],
        'pde_losses': [],
        'bc_losses': [],
        'l2_losses': []
            }

# 在

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
# 根据解析解获得正确的方向
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
    # xac_0[:, 0] *= s[0]
    # xac_0[:, 1] *= s[1]
    # x_test_0[:, 0] *= s[0]

   
    return u, s.cpu().detach().numpy(), corall.cpu().detach().numpy()
# %% active subspace
# 仅根据内点导数协方差矩阵更新
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
    # xac_0[:, 0] *= s[0]
    # xac_0[:, 1] *= s[1]
    # x_test_0[:, 0] *= s[0]

   
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

# dataset6, x0, x1, u6,corall = get_direction(model, dataset5)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset6, opt="LBFGS", steps=300);

# dataset7, x0, x1, u7,corall = get_direction(model, dataset6)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset7, opt="LBFGS", steps=300);

# dataset8, x0, x1, u8,corall = get_direction(model, dataset7)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset8, opt="LBFGS", steps=300);

#sys.exit()

# %% 获得不同来源的数据，包括准确解，原始模型解，新模型解

# 定义横纵坐标的范围
x_min, x_max = -1, 1
y_min, y_max = -1, 1
x_steps, y_steps = 100, 100

# 使用torch.linspace生成横纵坐标的值
x_coords = torch.linspace(x_min, x_max, x_steps)
y_coords = torch.linspace(y_min, y_max, y_steps)

# 使用torch.meshgrid生成坐标网格
x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

# 将网格堆叠起来形成最终的坐标张量
xgrid = torch.stack((x_grid, y_grid), dim=-1)

x0 = xgrid.view(-1, 2)  # 或者使用 x0_flat = x0.reshape(-1, 2)

# 精确解
y0 = sol_fun(x0)

# 原始模型解
y1 = model.cpu()(x0.cpu())


# 变换后变量x5
x1 = x0.cpu() @ u1.cpu().clone() 
x2 = x1.cpu() @ u2.cpu().clone() 
x3 = x2.cpu() @ u3.cpu().clone() 
x4 = x3.cpu() @ u4.cpu().clone() 
x5 = x4.cpu() @ u5.cpu().clone() 

# 获得变换矩阵
uu1 = u1.cpu().clone()
uu2 = u1.cpu().clone()@u2.cpu().clone()
uu3 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()
uu4 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()
uu5 = u1.cpu().clone()@u2.cpu().clone()@u3.cpu().clone()@u4.cpu().clone()@u5.cpu().clone()
# 新模型解
y2 = model1.cpu()(x1.cpu())


# %% 绘制图片

# 定义网格的边界
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

# 创建网格
X, Y = np.meshgrid(x, y)

# 确保y0, y1, y2都是numpy数组，并且与网格大小相匹配
# 这里假设y0, y1, y2已经是通过某种方式计算得到的，并且它们的大小是(100, 100)
y0_np = y0.detach().numpy().reshape(100, 100)
y1_np = y1.detach().numpy().reshape(100, 100)
y2_np = y2.detach().numpy().reshape(100, 100)

# 计算所有数据的最小值和最大值以确保颜色条范围一致
vmin = min(y0_np.min(), y1_np.min(), y2_np.min())
vmax = max(y0_np.max(), y1_np.max(), y2_np.max())

# 创建一个图形和三个子图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制y0的云图
im0 = axs[0].pcolormesh(X, Y, y0_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('True Solution y0')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# 绘制y1的云图
im1 = axs[1].pcolormesh(X, Y, y1_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model Solution y1')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

# 绘制y2的云图
im2 = axs[2].pcolormesh(X, Y, y2_np, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[2].set_title('Transformed Model Solution y2')
axs[2].set_xlabel('x0_1')
axs[2].set_ylabel('x0_2')

# 显示颜色条
#cbar = fig.colorbar(im0, ax=axs, label='Solution Value')

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()


# %% 残差对比

error1 = abs(y0_np-y1_np)
error2 = abs(y0_np-y2_np)
vmin = min(error1.min(), error2.min())
vmax = max(error1.max(), error2.max())

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# 绘制y1残差的云图
im0 = axs[0].pcolormesh(X, Y, abs(y0_np-y1_np), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('Model error y1')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# 绘制y2的云图
im1 = axs[1].pcolormesh(X, Y, abs(y0_np-y2_np), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model error y2')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

plt.show()


sys.exit()


# %%  plt the contour of function
# 需要修改选取的model,自变量x和fig的名字
y2 = model.cpu()(x0.cpu())
fig_name = 'result/solving_Laplace1_results_model0_value.png';


y2_np = y2.detach().numpy().reshape(100, 100)
# 创建一个图形和三个子图
fig, axs = plt.subplots(1, 1, figsize=(6, 5))
im2 = axs.pcolormesh(X, Y, y2_np, cmap='YlOrBr', shading='auto', vmin=-1, vmax=1)

cbar = fig.colorbar(im2, ax=axs)
cbar.ax.tick_params(labelsize=20)
#cbar.set_label('Color scale label')
# 设置纵轴范围
plt.ylim(-1, 1)  # 确保横轴范围为 0.1 到 1
# 设置纵轴的刻度为每隔 0.2 一个
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.xticks(np.arange(-1, 1.1, 0.5))
# 设置小刻度，每两个大刻度之间一个小刻度
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))

plt.tick_params(axis='y', direction='in')
plt.tick_params(axis='x', direction='in')
# 设置刻度的粗细
plt.tick_params(axis='both', which='major', width=2, length=8)  # 设置大刻度的粗细
plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')  # 设置小刻度的粗细并向内
plt.tick_params(axis='both', labelsize=25)  # 将坐标轴刻度标签的字体大小设置为30磅
#axs.set_title('Transformed Model Solution y2')
axs.set_xlabel(r'$x_1$',fontsize=35)
axs.set_ylabel(r'$x_2$',fontsize=35)
plt.tick_params(axis='x', which='both', pad=10)  # X轴刻度值往下挪10个像素
# 调整子图间距
plt.tight_layout()
plt.savefig(fig_name, dpi=600, bbox_inches='tight')

# 显示图形
plt.show()



# %%  plt the contour of function error
# 需要修改选取的model,自变量x和fig的名字
y2 = model.cpu()(x0.cpu())
fig_name = 'result/pde_solving_results_model0_error.png';

y2_np = y2.detach().numpy().reshape(100, 100)
error = abs(y0_np-y2_np)


# 创建一个图形和三个子图
fig, axs = plt.subplots(1, 1, figsize=(6, 5))
im2 = axs.pcolormesh(X, Y, error, cmap='YlOrBr', shading='gouraud', vmin=0, vmax=0.2)

cbar = fig.colorbar(im2, ax=axs)
cbar.ax.tick_params(labelsize=20)
#cbar.set_label('Color scale label')
# 设置纵轴范围
plt.ylim(-1, 1)  # 确保横轴范围为 0.1 到 1
# 设置纵轴的刻度为每隔 0.2 一个
plt.yticks(np.arange(-1, 1.1, 0.5))
plt.xticks(np.arange(-1, 1.1, 0.5))
# 设置小刻度，每两个大刻度之间一个小刻度
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))

plt.tick_params(axis='y', direction='in')
plt.tick_params(axis='x', direction='in')
# 设置刻度的粗细
plt.tick_params(axis='both', which='major', width=2, length=8)  # 设置大刻度的粗细
plt.tick_params(axis='both', which='minor', width=2, length=4, direction='in')  # 设置小刻度的粗细并向内
plt.tick_params(axis='both', labelsize=23)  # 将坐标轴刻度标签的字体大小设置为30磅
#axs.set_title('Transformed Model Solution y2')
axs.set_xlabel(r'$x_1$',fontsize=35)
axs.set_ylabel(r'$x_2$',fontsize=35)
plt.tick_params(axis='x', which='both', pad=10)  # X轴刻度值往下挪10个像素
# 调整子图间距
plt.tight_layout()
plt.savefig(fig_name, dpi=600, bbox_inches='tight')

# 显示图形
plt.show()


# %% plt the model structure
model1.plot()
figname = 'result/solving_Laplace1_model1.png'
# 保存图形为 600 dpi
plt.savefig(figname, dpi=600, bbox_inches='tight')

# 如果你想要显示图片，可以调用 plt.show()
plt.show()

# %% plt the error
# 假设 results_model1 是一个字典，包含训练和测试损失
from matplotlib import rc
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator, AutoMinorLocator
rc('font',**{'family':'serif','serif':['Times New Roman']})
train_loss1 = np.array(results_model['l2_losses'])
test_loss1 = np.array(results_model1['l2_losses'])
figname = 'result/solving_Laplace_Loss.png'

# 创建图形
plt.figure(figsize=(10, 5))

# 绘制训练损失，使用实线，颜色为较深的蓝色，线条粗细为2
plt.plot(train_loss1, label='KAN', linestyle='-', marker = 'o', color='blue', markevery=17, markersize=10, linewidth=2)

# 绘制测试损失，使用虚线，颜色为较浅的蓝色，线条粗细为2
plt.plot(test_loss1, label='ASM-KAN', linestyle='--', marker = 'v', color='DodgerBlue', markevery=20, markersize=10, linewidth=2)

# 添加图例，并设置字体大小
#plt.legend(fontsize=25)
plt.legend(fontsize=25, framealpha=0.5, facecolor='gray')

# 添加轴标签，并设置字体大小
# 设置横轴范围
plt.xlim(-10, 200)  # 确保横轴范围为 0.1 到 1


# 设置纵轴范围
#plt.ylim(-0.005, 10)  # 确保横轴范围为 0.1 到 1

# 设置y轴为对数坐标
plt.yscale('log')
plt.ylim(-1+10**-5, 1+10**0) # 需要根据不同的算例进行调整
#plt.yticks([10**-5,10**-4,10**-3,10**-2,10**-1], [r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],fontsize=30)

# 设置纵轴的刻度为每隔 0.1 一个
#plt.yticks(np.arange(0.0, 10, 0.02))

# 添加小刻度
# 设置小刻度，每两个大刻度之间一个小刻度
#plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
#plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))
y_ticks = [10**i for i in range(-5, 1)]  # 从 10^-5 到 10^0
plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))

# 设置y轴刻度样式，确保右边也有小刻度
plt.tick_params(axis='y', direction='in', right=True, labelright=False, which='both')

# 设置横轴的刻度为每隔 50 一个
plt.xticks(np.arange(0, 201, 50))

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

