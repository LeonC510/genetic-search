import numpy as np

def error(a, b):
   # defines the function from here https://www.youtube.com/watch?v=1i8muvzZkPw
    func = 3*(1 - a)**2*np.exp(-a**2-(b+1)**2) - 10*(a/5-a**3-b**5)*np.exp(-a**2-b**2) - (1/3)*np.exp(-b**2-(a+1)**2)
    return -func + 7.8