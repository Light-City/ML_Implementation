import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt
# 原始数据
data = [[3, 3], [4, 3], [1, 1]]
# shape=(3,2)
X = np.array(data)
print(X)
# shape=(3,1)
y = np.array([1, 1, -1])

# 设a=1,b=0,w为shape=(2,1)
a=1
# 初始化为0
w=np.zeros((2,1))
print(w)
b=0

# 设定循环
flag = True
length = len(X)
print(length)
j = 1

# 误分类列表
errorpoint_list=[0]
# 权重列表
w_list=[0]
# 偏值列表
b_list=[0]
# 函数表达式列表
wb_list=[0]
while flag:
    count = 0
    print("第" + str(j) + "次纠正")
    for i in range(length):
        # 取出x的坐标点，shape=(1,2)x(2,1)=(1,1)
        # w*x+b运算
        wb = int(np.dot(X[i,:], w) + b)
        # 寻找错误点
        if wb * y[i] <= 0:
            w += (y[i]*a*X[i,:]).reshape(w.shape)
            b += a*y[i]
            count += 1
            print("x"+str(i+1)+"为误分类点")
            errorpoint_list.append((i+1))
            print(w)
            w_list.append((int(w[0][0]),int(w[1][0])))
            print(b)
            b_list.append(b)
            wb_function = str(int(w[0][0]))+"*x1+"+str(int(w[1][0]))+"*x2+("+str((b))+")"
            print(wb_function)
            wb_list.append(wb_function)
            break
    if count == 0:
        flag = False
    j+=1
# 最后被break掉的数据添加到各自列表
errorpoint_list.append(0)
w_list.append((int(w[0][0]),int(w[1][0])))
b_list.append(b)
wb_function = str(int(w[0][0]))+"*x1+"+str(int(w[1][0]))+"*x2+("+str((b))+")"
wb_list.append(wb_function)


# 可视化表2.1
pt = PrettyTable()
pt.add_column("迭代次数",np.linspace(0,8,9,dtype=int))
pt.add_column("误分类点",errorpoint_list)
pt.add_column("w",w_list)
pt.add_column("b",b_list)
pt.add_column("w*x+b",wb_list)
print(pt)


# 可视化
x = np.linspace(0, 5, 200)
# 最终的函数表达式为w[0][0]*x+w[1][0]*y=0,推导后就是下面的式子
y = (-b - w[0][0] * x) / w[1][0]
plt.plot(x, y, color='r')
plt.scatter(X[:2, 0], X[:2, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[2:, 0], X[2:, 1], color='red', marker='x', label='Negative')
plt.xlabel('x')
plt.ylim(0,5)
plt.ylabel('y')
plt.legend()
plt.title('PLA')
plt.savefig('pla.png', dpi=75)
plt.show()

