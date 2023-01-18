#!/usr/bin/env python
# coding: utf-8

# # Multiple Minima Vs Initial GuessAnd Advance Functions 

# # $$g(x) = x^4 - 4x^2 + 5$$

# In[34]:


import matplotlib.pyplot as plt
import numpy as np 


get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


#Make some data

x_2 = np.linspace(-2, 2, 1000)



def g(x):
    return x**4 - 4*x**2 + 5



def dg(x):
    return 4*x**3 - 8*x 


# In[36]:


#Chart One Cost Function
plt.figure(figsize= [15,5])


plt.subplot(1,2,1)


plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)

plt.title('Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), color='blue', linewidth=3)


#second chart Derivative

plt.subplot(1,2,2)


plt.title('Slope Of The Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 3)
plt.ylim(-3, 6)

plt.plot(x_2, dg(x_2), color='skyblue', linewidth=3)





plt.show()


# # Gradient Descent As Python Function 

# In[37]:


def gradient_descent(derivative_func, initial_guess,multiplier=0.02, precision= 0.01):


    new_x = initial_guess        #(starting point)
    x_list = [new_x]
    slope_list= [derivative_func(new_x)]

    for n in range(500):
        previous_x= new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient

        step_size = abs(new_x - previous_x)
       # print(step_size)  

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))


        if step_size < precision:
            break

    return new_x, x_list, slope_list


# In[38]:


local_min, list_x, deriv_list = gradient_descent(dg, 0.05, 0.02, -0.5)
print('local minima occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[39]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=-0.1)
print('local minima occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[44]:


#calling gradient descent function 
local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= 0.1)


#plot function and derivative side by side 
plt.figure(figsize= [15,5])


#chart one cost function 

plt.subplot(1,2,1)


plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)

plt.title('Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), color='blue', linewidth=3, alpha=0.6)
plt.scatter(list_x, g(np.array(list_x)), color = 'red', s=100, alpha=0.6)


#second chart Derivative

plt.subplot(1,2,2)


plt.title('Slope Of The Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-6, 8)

plt.plot(x_2, dg(x_2), color='skyblue', linewidth=3, alpha=0.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=0.6)



plt.show()


# In[ ]:




