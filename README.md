# Todo

1. 阅读AirSim文档`https://frendowu.github.io/AirSim-docs-zh`
2. 穿圈后，有速度突变，姿态改变过快，可能现实环境会炸机


# Task Record

2023年4月5日10:10:13
---
1. 学习了第二次宣讲
2. 配置了运行虚拟环境
3. 简单交流

2023年4月7日17:17:45
---
1. 注释掉了`waittime` 和`cv2.imshow`
2. 现状：192s 5collisons @Eternallightning 
3. 修改了``run.bat`分辨率

2023年4月7日23:13:05
---
1. 在 @Alex Cui 协助下，小组完整阅读了所有代码，了解各函数的接口返回值和实现的function
2. @LLAYUAN 速通Python
3. @Etenallightning 共享了Remote Desktop，拯救小组的硬件于水火之中
4. @Yiliu1412 摸鱼了，进行了进度跟进

2023年4月16日15:48:03
---
1. `move` 不传参 z轴偏高
2. moveby 往下飞 z轴偏高
3. yawmode 在偏航角是不是零度就不调整
4. 重构了找圆心算法
5. 移动算法没有改动
6. [8字爆算移动](https://zhuanlan.zhihu.com/p/485796378)
7. 现状：86.7s 0 collisons @LLAYUAN
    - 降落时间
    - 尝试把过飞的情况通过提前减速处理，但有概率找不到圆，注释掉了未使用
    - 强行停下来，速度降下来会z轴偏高

2023年4月16日21:25:33
---
- 完全重写了代码
    
2023年4月23日22:24:30
---
1. 最后review了代码
2. 打包代码


ALL LAST
---
