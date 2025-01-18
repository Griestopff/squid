import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei laden
data = pd.read_csv('speedup_results_size.csv')

# Plot erstellen
plt.figure(figsize=(10, 6))

# Speedups plotten
plt.plot(data['SIZE'], data['Speedup_gnu_parallel'], marker='o', color='g', label='gnu_parallel::sort')
plt.plot(data['SIZE'], data['Speedup_min_max_quicksort'], marker='o', color='b', label='min_max_quicksort')


# Plot-Details
plt.xlabel('Array size')
plt.ylabel('Average Speedup (of five rounds)')
plt.title('Speedup min_max_quicksort and __gnu_parallel::sort (in comparison to std::sort)')
plt.legend(loc='upper left')  # Position der Legende
plt.grid(True)
plt.xscale('log')

# Plot anzeigen
plt.show()

