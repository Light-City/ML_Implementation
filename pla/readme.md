
![](http://p20tr36iw.bkt.clouddn.com/pla.png)


# 可视化PLA算法


## 0.说在前面

之前[Perceptron Learning Algorithm](https://mp.weixin.qq.com/s/NVwUu2ZzKvhhwfUAAgRVEg)这篇文章详细讲了感知机PLA算法，前两天买了本统计学习方法，今天早上看了两章，其中第二章就是这个PLA，跟李老师的课程讲的基本一致，本节主要通过python实现这个感知机算法，并通过matlibplot可视化图形，以及终端打印出下图结果！

![](http://p20tr36iw.bkt.clouddn.com/LPA_LEA.png)





## 1.实现

原理请参考网上教程，或者我在前言的文章，再或者统计学习方法书上算法。

【**导包**】

分别用于矩阵，表格数据打印，数据可视化。

```python
import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt
```

【**初始化**】

```python
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
```

【**算法实现**】

```python
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
```

【**可视化表**】

```python
# 可视化表2.1
pt = PrettyTable()
pt.add_column("迭代次数",np.linspace(0,8,9,dtype=int))
pt.add_column("误分类点",errorpoint_list)
pt.add_column("w",w_list)
pt.add_column("b",b_list)
pt.add_column("w*x+b",wb_list)
print(pt)
```

![](http://p20tr36iw.bkt.clouddn.com/pla_termi.png)

【**最终结果可视化**】

```python
# 可视化
x = np.linspace(0, 7, 200)
# 最终的函数表达式为w[0][0]*x+w[1][0]*y=0,推导后就是下面的式子
y = (-b - w[0][0] * x) / w[1][0]
plt.plot(x, y, color='r')
plt.scatter(X[:2, 0], X[:2, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[2:, 0], X[2:, 1], color='red', marker='x', label='Negative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('PLA')
plt.savefig('pla.png', dpi=75)
plt.show()
```

![](http://p20tr36iw.bkt.clouddn.com/pla.png)


## 2.作者的话
有关机器学习的更多内容，请关注下面个人公众号
![](http://p20tr36iw.bkt.clouddn.com/wechat.jpg)