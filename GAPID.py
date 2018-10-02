import numpy as np
import random
import math
import copy
import vrep                  #V-rep's library
import sys
import time


class Ind():
    def __init__(self):
        self.fitness = 0
        self.x = np.zeros(35)   #基因位数
        self.place = 0
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0

def Cal_fit(x, upper, lower):   #计算适应度值函数
    summ=0
    Temp1 = 0
    for i in range(10):
        Temp1 += x[i] * math.pow(2, i)
    Temp2 = 0
    for i in range(10, 25, 1):
        Temp2 += math.pow(2, i - 10) * x[i]
    Temp3 = 0
    for i in range(25, 35, 1):#解码
        Temp3 += math.pow(2, i - 25) * x[i]
    x1 = lower[0] + Temp1 * (upper[0] - lower[0])/(math.pow(2, 10) - 1)
    x2 = lower[1] + Temp2 * (upper[1] - lower[1])/(math.pow(2, 15) - 1)
    x3 = lower[2] + Temp3 * (upper[2] - lower[2])/(math.pow(2, 10) - 1)
    if x1 > upper[0]:
        x1 = random.uniform(lower[0], upper[0])
    if x2 > upper[1]:
        x2 = random.uniform(lower[1], upper[1])
    if x3 > upper[2]:
        x3 = random.uniform(lower[2], upper[2])
    summ=run(x1,x2,x3)
    return summ
def Init(G, upper, lower, Pop):    #初始化函数
    for i in range(Pop):
        for j in range(35):
            G[i].x[j] = random.randint(0, 1)
        G[i].fitness = Cal_fit(G[i].x, upper, lower)
        print ('G[i].fitness',G[i].fitness)
       # G[i].fitness=summ
        G[i].place = i
def Find_Best(G, Pop):#选优
    Temp = copy.deepcopy(G[0])
    for i in range(1, Pop, 1):
        if G[i].fitness < Temp.fitness:
            Temp = copy.deepcopy(G[i])
    return Temp

def Selection(G, Gparent, Pop, Ppool):    #选择函数
    fit_sum = np.zeros(Pop)
    fit_sum[0] = G[0].fitness
    for i in range(1, Pop, 1):
        fit_sum[i] = G[i].fitness + fit_sum[i - 1]
    fit_sum = fit_sum/fit_sum.max()
    for i in range(Ppool):
        rate = random.random()
        Gparent[i] = copy.deepcopy(G[np.where(fit_sum > rate)[0][0]])

def Cross_and_Mutation(Gparent, Gchild, Pc, Pm, upper, lower, Pop, Ppool):  #交叉和变异
    for i in range(Ppool):
        place = random.sample([_ for _ in range(Ppool)], 2)
        parent1 = copy.deepcopy(Gparent[place[0]])
        parent2 = copy.deepcopy(Gparent[place[1]])
        parent3 = copy.deepcopy(parent2)
        if random.random() < Pc:
            num = random.sample([_ for _ in range(1, 34, 1)], 2)
            num.sort()
            if random.random() < 0.5:
                for j in range(num[0], num[1], 1):
                    parent2.x[j] = parent1.x[j]
            else:
                for j in range(0, num[0], 1):
                    parent2.x[j] = parent1.x[j]
                for j in range(num[1], 35, 1):
                    parent2.x[j] = parent1.x[j]
            num = random.sample([_ for _ in range(1, 34, 1)], 2)
            num.sort()
            num.sort()
            if random.random() < 0.5:
                for j in range(num[0], num[1], 1):
                    parent1.x[j] = parent3.x[j]
            else:
                for j in range(0, num[0], 1):
                    parent1.x[j] = parent3.x[j]
                for j in range(num[1], 35, 1):
                    parent1.x[j] = parent3.x[j]
        for j in range(35):
            if random.random() < Pm:
                parent1.x[j] = (parent1.x[j] + 1) % 2
            if random.random() < Pm:
                parent2.x[j] = (parent2.x[j] + 1) % 2

        parent1.fitness = Cal_fit(parent1.x, upper, lower)
        parent2.fitness = Cal_fit(parent2.x, upper, lower)
        Gchild[2 * i] = copy.deepcopy(parent1)
        Gchild[2 * i + 1] = copy.deepcopy(parent2)

def Choose_next(G, Gchild, Gsum, Pop):    #选择下一代函数
    for i in range(Pop):
        Gsum[i] = copy.deepcopy(G[i])
        Gsum[2 * i + 1] = copy.deepcopy(Gchild[i])
    Gsum = sorted(Gsum, key = lambda x: x.fitness, reverse = True)
    for i in range(Pop):
        G[i] = copy.deepcopy(Gsum[i])
        G[i].place = i

def Decode(x):            #解码函数
    Temp1 = 0
    for i in range(10):
        Temp1 += x[i] * math.pow(2, i)
    Temp2 = 0
    for i in range(10, 25, 1):
        Temp2 += math.pow(2, i - 10) * x[i]
    Temp3 = 0
    for i in range(25, 35, 1):
        Temp3 += math.pow(2, i - 25) * x[i]
    x1 = lower[0] + Temp1 * (upper[0] - lower[0]) / (math.pow(2, 10) - 1)
    x2 = lower[1] + Temp2 * (upper[1] - lower[1]) / (math.pow(2, 15) - 1)
    x3 = lower[2] + Temp3 * (upper[2] - lower[2]) / (math.pow(2, 10) - 1)
    if x1 > upper[0]:
        x1 = random.uniform(lower[0], upper[0])
    if x2 > upper[1]:
        x2 = random.uniform(lower[1], upper[1])
    if x3> upper[2]:
        x3 = random.uniform(lower[2], upper[2])
    return x1,x2,x3

def Self_Learn(Best, upper, lower, sPm, sLearn):  #自学习操作
    num = 0
    Temp = copy.deepcopy(Best)
    while True:
        num += 1
        for j in range(35):
            if random.random() < sPm:
                Temp.x[j] = (Temp.x[j] + 1)%2
        Temp.fitness = Cal_fit(Temp.x, upper, lower)
        if Temp.fitness < Best.fitness:
            Best = copy.deepcopy(Temp)
            num = 0
        if num > sLearn:
            break
    return Best



def convertsensor(a): #Function which is going to convert sensors values
    if a >0.95:        #When value is bigger than 0.95 function returns 0
        return 0.0      #When it is seeing line then it return 1.
    else:
        return 1.0       
vrep.simxFinish(-1) #It is closing all open connections with VREP
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:  #It is checking if connection is successful
    print ('Connected to remote API server')   
else:
    print( 'Connection not successful')
    sys.exit('Could not connect')

#Getting motor handles
errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,"left_joint",vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,"right_joint",vrep.simx_opmode_oneshot_wait)
sensor_h=[] #handles list
sensor_val=[] #Sensor value list
#Getting sensor handles list
for x in range(0,7):
        errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'line_sensor'+str(x),vrep.simx_opmode_oneshot_wait)
        sensor_h.append(sensor_handle) #It is adding sensor handle values
        errorCode,detectionstate, sensorreadingvalue=vrep.simxReadVisionSensor(clientID,sensor_h[x],vrep.simx_opmode_streaming)
        sensor_val.append(1.0) #It is adding 1.0 to fill the sensor values on the list. In the while loop it is going to overwrite the values
maxkiirus = 15 #Maximal speed
previouserror =0 #previous error
error = 0 #error
integral = 0 # PID values before while loop
derivative = 0
proportional = 0
time.sleep(1)
P=0.3;I=0.0003;D=0.5;
t = time.time() #It is saving the time which is now
summ=0;
def run(x1,x2,x3): #PID仿真函数
    k=0;
    viivitus=0;
    sensor_val[6]=0;
    while (sensor_val[6]<0.95):
        if k==0:            
            maxkiirus = 15 #Maximal speed
            previouserror =0 #previous error
            error = 0 #error
            integral = 0 # PID values before while loop
            derivative = 0
            proportional = 0
            time.sleep(1)
            t = time.time() #It is saving the time which is now
            summ=0;
        
        summa = 0  
        andur = 0	
        for x in range(0,6):
            errorCode,detectionstate, sensorreadingvalue=vrep.simxReadVisionSensor(clientID,sensor_h[x],vrep.simx_opmode_buffer)
            sensor_val[x]=sensorreadingvalue[1][0] #It is overwriting the sensor values
            andur = andur+convertsensor(sensor_val[x]) #It is adding the sensor values by each other
            summa=summa +convertsensor(sensor_val[x])*x*10.0 #Calculating the sum where the robots position is        
    
        if andur == 0 and previouserror >0: #When no sensors doesn't see and previous error is bigger than zero
            error = maxkiirus # Error is going to be maximal speed
        elif andur == 0 and previouserror<0: #When no sensors doesn't see and previous error is lower than zero
            error =-maxkiirus # Error is going to be -maximal speed
        else:
            if andur == 0: # When sensor doesn't see
                positsioon = proportional # Position is going to be proportional error
            else:
                positsioon =summa/andur # otherwise position is the division between sum and sensor values
            #print( "Positsioon on :", positsioon)
            proportional = positsioon -25 # Proportional error
            summ=summ+proportional*proportional
            
            
            #print( "viivitus :",viivitus)
            integral = integral + proportional # Integral error
            if integral >2500: #When integral value is going to be too big
                integral =0    #integral is set to zero
            elif integral <-2500:
                integral =0
         #   print "Integraal on :", integral
            derivative = proportional - previouserror # Derivative error - in reality this has no effect.
         #   print "Derivatiivne on :", derivative
            previouserror = proportional #It is remembering previous error
            error =  proportional*x1 + integral*x2 +derivative*x3 #Calculating overall error using PID.
         
          #  print "viivitus on: ", viivitus
        #When position (error) is to big/small, then the speed is set max/-max.
        if error > maxkiirus: #When error is bigger than maximal speed,
           error = maxkiirus  #error is set to max speed
        elif error < -maxkiirus: # It is because the wheels shouldn't rotate backwards.
           error = -maxkiirus
        if error <0: # In here it is calculating the speed of the wheels
            omega_right=maxkiirus+error
            omega_left=maxkiirus
        else:
            omega_right=maxkiirus
            omega_left=maxkiirus-error    
      
        errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,omega_left, vrep.simx_opmode_streaming)
        errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,omega_right, vrep.simx_opmode_streaming)
        k=k+1;
        errorCode,detectionstate, sensorreadingvalue=vrep.simxReadVisionSensor(clientID,sensor_h[6],vrep.simx_opmode_buffer)
        sensor_val[6]=sensorreadingvalue[1][0]
        viivitus = round((time.time()-t),5) #calculating delay time
        if viivitus<3:   #3秒内不检测终点位置
            sensor_val[6]=0
 #   print(" hfhg  ",sensor_val[6])    
 #   print(" viivitus ",viivitus)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
    return summ
if __name__ == '__main__':#遗传迭代开始
    upper = [1,0.01,2]
    lower = [0,0,0]
    Pop = 40
    Ppool =20
    G_max =20          #迭代20代
    Pc = 0.8
    Pm = 0.1
    sPm = 0.05
    sLearn = 20
    G = np.array([Ind() for _ in range(Pop)])
    Gparent = np.array([Ind() for _  in range(Ppool)])
    Gchild = np.array([Ind() for _ in range(Pop)])
    Gsum = np.array([Ind() for _ in range(Pop * 2)])
    Init(G, upper, lower, Pop)       #初始化
    print (" 1  ")
    Best = Find_Best(G, Pop)
    print (" 2  ")
    for k in range(G_max):
        print (" kk  ",k)
        Selection(G, Gparent, Pop, Ppool)       #使用轮盘赌方法选择其中50%为父代
        print (" 1...1  ")
        Cross_and_Mutation(Gparent, Gchild, Pc, Pm, upper, lower, Pop, Ppool)  #交叉和变异生成子代
        print (" 1...2  ")
        Choose_next(G, Gchild, Gsum, Pop)       #选择出父代和子代中较优秀的个体
        print (" 1...3  ")
        Cbest = Find_Best(G, Pop)
        print (" 1...4  ")
        if Best.fitness > Cbest.fitness:
            Best = copy.deepcopy(Cbest)        #跟新最优解
        else:
            G[Cbest.place] = copy.deepcopy(Best)
        print (" 1...5  ")
        Best = Self_Learn(Best, upper, lower, sPm, sLearn)
        print(Best.fitness)
    x1, x2 ,x3= Decode(Best.x)
    print(Best.x)
    print([x1, x2,x3])           




















