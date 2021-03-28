import matplotlib.pyplot as plt
import math
import random
import json
import numpy 
class oscillator:
    def __init__(self, gamma, a, omega, alpha, time, velocity, iterations):
        self.gamma = gamma
        self.a = a
        self.omega = omega
        self.alpha = alpha
        self.time = time
        self.velocity = velocity
        self.iterations = iterations
        self.x_points = []
        self.y_points = []
    
    def iterate_underdamped(self):
        omega_1 = (self.omega**2 - self.gamma**2)**(1/2)
        i = 0;
        while i < self.time:
            x = math.e**(-self.gamma*i)*self.a*math.cos(omega_1*i - self.alpha)
            self.x_points.append(i)
            self.y_points.append(x/self.a)
            i += self.time/self.iterations
            
    def iterate_critically(self):
        i = 0;
        while i < self.time:
            x = math.e**(-self.gamma*i)*(self.a + (self.velocity + self.gamma*self.a)*i)
            self.x_points.append(i)
            self.y_points.append(x/self.a)
            i += self.time/self.iterations
            
''' def show_underdamped(self):
        print("debug1")
        self.iterate_underdamped()
        plt.plot(damped_oscillator.x_points[1:], damped_oscillator.y_points[1:])
        plt.plot(damped_oscillator.x_points[0:1], damped_oscillator.y_points[0:1])  
        plt.title("Underdamped Spring")
        plt.xlabel("time (seconds)")
        plt.ylabel(r"$\frac{X}{X_0}$")
        plt.show()
'''



'''  def show_critically(self):
        print("debug")
        self.iterate_critically()
        plt.plot(damped_oscillator.x_points[1:], damped_oscillator.y_points[1:])
        plt.plot(damped_oscillator.x_points[0:1], damped_oscillator.y_points[0:1])  
        plt.title("Critically Damped Spring")
        plt.xlabel("time (seconds)")
        plt.ylabel(r"$\frac{X}{X_0}$")
        plt.show()  '''

#Oscillator(Gamma Value (dampenning), A (maximum) value, Omega Value, Alpha Value, Time scale, Initial velocity, Iterations to plot)

'''if __name__ == "__main__":
    data = []
    for i in range(5):
        Omega = random.uniform(5,10)
        Alpha = random.uniform(5,10)
        Velocity = random.uniform(-1,1)
        if (i % 100) == 0:
            print(i)
        batch = []
        Gamma = random.uniform(0.5,1)
        print(Gamma, Omega, Alpha)
        damped_oscillator = oscillator(Gamma, 1, Omega, Alpha, 10, Velocity, 100)
        mylist = [damped_oscillator.show_critically,damped_oscillator.show_underdamped]
        #damped_oscillator.show_critically()
        random.choice(mylist)()
        batch = [damped_oscillator.x_points[0:50],damped_oscillator.y_points[0:50]]
        question = random.randint(0,100)
        data.append([batch,[damped_oscillator.x_points[question],damped_oscillator.y_points[question]]])
        
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)'''
