import numpy as np
import matplotlib.pyplot as plt

days = 30
steps = np.random.randint(2000, 16000, days)
calories = steps * 0.04 + np.random.normal(200, 50, days)
heart_rate = np.random.randint(60, 180, days)

fitness_data = np.column_stack((steps, calories, heart_rate))

print("Fitness Data (Steps, Calories, Heart Rate):\n", fitness_data[:5])

avg_steps = np.mean(steps)
avg_calories = np.mean(calories)
avg_heart_rate = np.mean(heart_rate)

max_steps_day = np.argmax(steps) + 1
min_steps_day = np.argmin(steps) + 1

print(f"Average Steps: {avg_steps:.2f}")
print(f"Average Calories Burned: {avg_calories:.2f}")
print(f"Average Heart Rate: {avg_heart_rate:.2f}")
print(f"Day with Maximum Steps: Day {max_steps_day}")
print(f"Day with Minimum Steps: Day {min_steps_day}")

week = fitness_data[:28].reshape(4, 7, 3)

weekly_avg = np.mean(week, axis=1)

print("\nWeekly Average (Steps, Calories, Heart Rate):\n", weekly_avg)

corr_steps_calories = np.corrcoef(steps, calories)[0, 1]
corr_steps_heart_rate = np.corrcoef(steps, heart_rate)[0, 1]

print(f"\nCorrelation between Steps and Calories: {corr_steps_calories:.2f}")
print(f"Correlation between Steps and Heart Rate: {corr_steps_heart_rate:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(steps, label='Steps')
plt.plot(calories, label='Calories Burned')
plt.plot(heart_rate, label='Heart Rate')
plt.title('Daily Fitness Trends')
plt.xlabel('Day')
plt.ylabel('Metrics')
plt.legend()
plt.show()

weeks_avg_steps = weekly_avg[:, 0]
plt.bar(range(1, 5), weeks_avg_steps, color='skyblue')
plt.title('Weekly Average Steps')
plt.xlabel('Week')
plt.ylabel('Average Steps')
plt.show()

plt.scatter(steps, calories, alpha=0.7, color='purple')
plt.title('Correlation: Steps vs Calories')
plt.xlabel('Steps')
plt.ylabel('Calories')
plt.show()