import numpy as np
import time
import matplotlib.pyplot as plt

# Fungsi yang akan diintegrasikan
def f(x):
    return 4 / (1 + x**2)

# Metode integrasi Riemann
def riemann_integration(N):
    a = 0
    b = 1
    h = (b - a) / N
    integral_sum = 0
    
    for i in range(N):
        x_i = a + (i + 0.5) * h
        integral_sum += f(x_i)
        
    integral = h * integral_sum
    return integral

# Nilai referensi pi
pi_ref = 3.14159265358979323846

# Variasi nilai N
N_values = [10, 100, 1000, 10000]

# Hasil dan waktu eksekusi
results = []
rms_errors = []
execution_times = []

for N in N_values:
    start_time = time.time()
    pi_approx = riemann_integration(N)
    end_time = time.time()
    
    execution_time = end_time - start_time
    rms_error = np.sqrt((pi_ref - pi_approx)**2)
    
    results.append(pi_approx)
    rms_errors.append(rms_error)
    execution_times.append(execution_time)
    
    print(f'N = {N}, Approximated pi = {pi_approx}, RMS Error = {rms_error}, Execution Time = {execution_time} seconds')

# Plot hasil RMS Error
plt.figure(figsize=(10, 5))
plt.plot(N_values, rms_errors, marker='o')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('RMS Error')
plt.title('RMS Error vs N')
plt.grid(True)
plt.show()

# Plot hasil Execution Time
plt.figure(figsize=(10, 5))
plt.plot(N_values, execution_times, marker='o')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs N')
plt.grid(True)
plt.show()
