import matplotlib.pyplot as plt
import math


class oscillator:
    def __init__(self, gamma, a, omega, alpha, time, iterations):
        self.gamma = gamma
        self.a = a
        self.omega = omega
        self.alpha = alpha
        self.time = time
        self.iterations = iterations
        self.x_points = []
        self.y_points = []
    
    def iterate(self):
        omega_1 = (self.omega**2 - self.gamma**2)**(1/2)
        i = 0;
        while i < self.time:
            x = math.e**(-self.gamma*i)*self.a*math.cos(omega_1*i - self.alpha)
            self.x_points.append(i)
            self.y_points.append(x/self.a)
            i += self.time/self.iterations
    def show(self):
        self.iterate()
        plt.plot(damped_oscillator.x_points, damped_oscillator.y_points)
        plt.xlabel("time (seconds)")
        plt.ylabel(r"$\frac{X}{X_0}$")
        plt.show()	
        
#Oscillator(Gamma Value (dampenning), A (maximum) value, Omega Value, Alpha Value, Time scale, Iterations to plot)
damped_oscillator = oscillator(1, 5, 10, 0, 10, 1000)
damped_oscillator.show()


