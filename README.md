# GAPID

1、利用遗传算法优化循迹小车PID参数

2、算法采用三元函数最小值优化遗传算法为主函数，调用PID控制子函数完成仿真

3、将PID三个参数看做一个染色体的三个基因（程序中用35位二进制表示，三参数分别占用10、15、10位二进制）

4、遗传算法控制量为（小车运动一圈）位置误差的平方和

5、一圈终点由line_sensor6感知
