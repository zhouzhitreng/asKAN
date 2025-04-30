#!/usr/bin/env python
# coding: utf-8

# # Example 1: Function Fitting
# 
# In this example, we will cover how to leverage grid refinement to maximimze KANs' ability to fit functions

# intialize model and create dataset

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

# 前几次迭代的time和loss变化
timechange = [81.11034560203552, 171.96951460838318,262.63033533096313,355.7091255187988,446.70072412490845,537.3332760334015]
losschange = [0.265,0.0509,0.0246,0.0342,0.0108,0.0106]

    

# initialize KAN with G=3
width=[2,5,1]
grid=5
k = 3
seed=1

model = KAN(width, grid, k, seed, device=device)

# create dataset
#f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
# f = lambda x: (torch.cos(12*x[:,[0]]**2 + 12*x[:,[1]]**2))

# function 1
f = lambda x: torch.exp(torch.cos(3*torch.pi*(((1.2*x[:,[0]])+0.6*x[:,[1]]))))
width=[2,5,1]
grid=5
k = 3
seed=1  
model = KAN(width, grid, k, seed, device=device)   

# function 2
# f = lambda x: torch.tanh(2*torch.pi*(x[:,[0]]*x[:,[1]]+x[:,[0]]+x[:,[1]]))
# width=[2,5,1]
# grid=5
# k = 3
# seed=1  
# model = KAN(width, grid, k, seed, device=device)        
    #torch.cos(2*torch.pi*(((0.5*x[:,[0]])+1.3*x[:,[1]])))
#f = lambda x: (torch.sin(3*torch.pi*((0.3*(x[:,[0]]+x[:,[1]])**2+0.7*x[:,[1]]**1))))

dataset = create_dataset(f, n_var=2, device=device)


# update the direction of variables


#dataset1 = 
# Train KAN (grid=3)
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
    # xac_0[:, 0] *= s[0]
    # xac_0[:, 1] *= s[1]
    # x_test_0[:, 0] *= s[0]
    # x_test_0[:, 1] *= s[1]
    
    
    dataset1 = copy.deepcopy(dataset)
    # xac_0.requires_grad = False
    # x_test_0.requires_grad = False
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


dataset3, x0, x1, u3,corall = get_direction(model2, dataset2)
model3 = KAN(width, grid, k, seed, device=device)
results_model3 = model3.fit(dataset3, opt="LBFGS", steps=200);
model3.plot()

end_time4 = time.time()

dataset4, x0, x1, u4,corall = get_direction(model3, dataset3)
model4 = KAN(width, grid, k, seed, device=device)
results_model4 = model4.fit(dataset4, opt="LBFGS", steps=200);
model4.plot()

end_time5 = time.time()

dataset5, x0, x1, u5,corall = get_direction(model4, dataset4)
model5 = KAN(width, grid, k, seed, device=device)
results_model5 = model5.fit(dataset5, opt="LBFGS", steps=200);
model5.plot()

end_time6 = time.time()
# dataset6, x0, x1, u6,corall = get_direction(model, dataset5)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset6, opt="LBFGS", steps=300);

# dataset7, x0, x1, u7,corall = get_direction(model, dataset6)
# m1odel = KAN(width, grid, k, seed, device=device)
# model.fit(dataset7, opt="LBFGS", steps=300);

# dataset8, x0, x1, u8,corall = get_direction(model, dataset7)
# model = KAN(width, grid, k, seed, device=device)
# model.fit(dataset8, opt="LBFGS", steps=300);

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
y0 = f(x0)

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
y2 = model2.cpu()(x2.cpu())


# %% PLT 

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
im0 = axs[0].pcolormesh(X, Y, y0_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('True Solution y0')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# 绘制y1的云图
im1 = axs[1].pcolormesh(X, Y, y1_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model Solution y1')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

# 绘制y2的云图
im2 = axs[2].pcolormesh(X, Y, y2_np, cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
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
im0 = axs[0].pcolormesh(X, Y, abs(y0_np-y1_np), cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[0].set_title('Model error y1')
axs[0].set_xlabel('x0_1')
axs[0].set_ylabel('x0_2')

# 绘制y2的云图
im1 = axs[1].pcolormesh(X, Y, abs(y0_np-y2_np), cmap='YlOrBr', shading='auto', vmin=vmin, vmax=vmax)
axs[1].set_title('Model error y2')
axs[1].set_xlabel('x0_1')
axs[1].set_ylabel('x0_2')

plt.show()


sys.exit()
# %% output the results
rmodel = results_model5; 
rmodel_name = 'result/fit_function1_results_model5';
namereg = rmodel_name + '_reg';
nametrain_loss = rmodel_name + '_train_loss';
nametest_loss  = rmodel_name + '_test_loss';
np.savetxt(namereg, np.array(rmodel['reg']), fmt='%d')
np.savetxt(nametrain_loss, np.array(rmodel['train_loss']), fmt='%d')
np.savetxt(nametest_loss, np.array(rmodel['test_loss']), fmt='%d')


# %%  plt the contour of function error
# 需要修改选取的model,自变量x和fig的名字
y2 = model5.cpu()(x5.cpu())
fig_name = 'result/fit_function1_results_model5_error.png';

y2_np = y2.detach().numpy().reshape(100, 100)
error = abs(y0_np-y2_np)


# 创建一个图形和三个子图
fig, axs = plt.subplots(1, 1, figsize=(6, 5))
im2 = axs.pcolormesh(X, Y, error, cmap='YlOrBr', shading='auto', vmin=0, vmax=0.2)

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

# %%  plt the contour of function
# 需要修改选取的model,自变量x和fig的名字
y2 = model.cpu()(x0.cpu())
fig_name = 'result/fit_function1_results_model0_value.png';


y2_np = y2.detach().numpy().reshape(100, 100)
# 创建一个图形和三个子图
fig, axs = plt.subplots(1, 1, figsize=(6, 5))
im2 = axs.pcolormesh(X, Y, y2_np, cmap='YlOrBr', shading='auto', vmin=0, vmax=4)

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
plt.tick_params(axis='both', labelsize=30)  # 将坐标轴刻度标签的字体大小设置为30磅
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
model.plot()
figname = 'result/fit_function1_model0.png'
# 保存图形为 600 dpi
plt.savefig(figname, dpi=600, bbox_inches='tight')

# 如果你想要显示图片，可以调用 plt.show()
plt.show()

# %% plt the error
# 假设 results_model1 是一个字典，包含训练和测试损失
train_loss1 = np.array(results_model2['train_loss'])
test_loss1 = np.array(results_model2['test_loss'])
figname = 'result/fit_function1_Loss2.png'

# 创建图形
plt.figure(figsize=(6, 5))

# 绘制训练损失，使用实线，颜色为较深的蓝色，线条粗细为2
plt.plot(train_loss1, label='Train Loss', linestyle='-', marker = 'o', color='blue', markevery=12, markersize=10, linewidth=2)

# 绘制测试损失，使用虚线，颜色为较浅的蓝色，线条粗细为2
plt.plot(test_loss1, label='Test Loss', linestyle='--', marker = 'v', color='DodgerBlue', markevery=15, markersize=10, linewidth=2)

# 添加图例，并设置字体大小
#plt.legend(fontsize=25)
plt.legend(fontsize=25, framealpha=0.5, facecolor='gray')

# 添加轴标签，并设置字体大小
# 设置横轴范围
plt.xlim(-10, 200)  # 确保横轴范围为 0.1 到 1


# 设置纵轴范围
plt.ylim(-0.1, 1)  # 确保横轴范围为 0.1 到 1

# 设置纵轴的刻度为每隔 0.1 一个
plt.yticks(np.arange(0.0, 1.1, 0.2))
# 添加小刻度
# 设置小刻度，每两个大刻度之间一个小刻度
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))

# 设置y轴刻度样式，确保右边也有小刻度
plt.tick_params(axis='y', direction='in', right=True, labelright=False, which='both')
plt.tick_params(axis='x', direction='in')

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


# %%绘制误差以及计算时间

# 配置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25  # 全局字号

# 数据准备
timechange = [81.11034560203552, 171.96951460838318, 262.63033533096313,
              355.7091255187988, 446.70072412490845, 537.3332760334015]
losschange = [0.265, 0.0509, 0.0246, 0.0342, 0.0108, 0.0106]
x = np.arange(len(timechange))  # 生成横坐标索引

# 创建画布和主纵轴（适当增大画布尺寸）
fig, ax1 = plt.subplots(figsize=(12, 7))

# ================== 主坐标轴设置 ==================
# 绘制时间折线图（左轴）
color = 'tab:blue'
ax1.set_xlabel('Level', fontweight='bold')  # 加粗标签
ax1.set_ylabel('Computation time (sec)', color=color, fontweight='bold')
line1, = ax1.plot(x, timechange, color=color, marker='o', markersize=10, 
                linewidth=3, label='Computation time')

# 刻度参数设置
ax1.tick_params(axis='y', labelcolor=color, labelsize=25)
ax1.tick_params(axis='x', which='both', labelsize=25)
ax1.grid(True, linestyle='--', alpha=0.6)

# ================== 次坐标轴设置 ==================
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Test error', color=color, fontweight='bold')
line2, = ax2.plot(x, losschange, color=color, marker='s', markersize=10,
                linewidth=3, linestyle='--', label='Test error')

# 刻度参数设置
ax2.tick_params(axis='y', labelcolor=color, labelsize=25)
ax2.set_ylim(0, 0.3)

# ================== 高级调整 ==================
# 统一x轴刻度
ax1.set_xticks(x)
ax1.set_xticklabels([f'{i+1}' for i in x])  # 显示具体层级名称

# 组合图例（调整位置和边距）
leg = ax1.legend(handles=[line1, line2], 
               loc='upper left',
               bbox_to_anchor=(0.15, 1.15),
               frameon=False,
               ncol=2)

# 优化布局
plt.subplots_adjust(top=0.85)  # 为顶部留出空间
fig.tight_layout()
plt.savefig('Figa_time', dpi=600, bbox_inches='tight')
# 显示图形
plt.show()

# In[ ]:




