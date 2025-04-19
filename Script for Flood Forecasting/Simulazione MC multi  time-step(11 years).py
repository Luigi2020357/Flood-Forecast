import numpy as np
import matplotlib.pyplot as plt


states = ['Low', 'Moderate', 'High', 'Very High']

# Nuova matrice di transizione
transition_matrix = np.array([
    [0.22222222, 0.22222222, 0.27777778, 0.27777778],  # Transizioni dallo stato 'Low'
    [0.10714286, 0.28571429, 0.35714286, 0.25],        # Transizioni dallo stato 'Moderate'
    [0.14285714, 0.46428571, 0.17857143, 0.21428571],  # Transizioni dallo stato 'High'
    [0.28, 0.16, 0.32, 0.24]                           # Transizioni dallo stato 'Very High'
])


def markov_simulation(transition_matrix, initial_state, steps):
    state = initial_state
    state_history = [state]
    
    for _ in range(steps):
        state = np.random.choice(states, p=transition_matrix[states.index(state)])
        state_history.append(state)
    
    return state_history 

# Parametri della simulazione
initial_state = 'Low'
years = 11  
steps_per_year = 12  # Simulazione mensile
total_steps = years * steps_per_year
# Esecuzione della simulazione
simulation_results = markov_simulation(transition_matrix, initial_state, total_steps)


plt.figure(figsize=(12, 6))
plt.plot(simulation_results, marker='o')
plt.title('Multi-year flood risk simulation')
plt.xlabel('Time step (month)')
plt.ylabel('Risk state')
plt.yticks(ticks=range(len(states)), labels=states)
plt.grid(True)
plt.show()
