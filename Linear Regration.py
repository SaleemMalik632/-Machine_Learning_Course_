import seaborn as sns
import matplotlib.pyplot as plt


# Sample data for line plot
x_line = [1, 2, 3, 4, 5]
y_line = [3, 5, 4, 6, 7]




# Create a line plot
plt.subplot(2, 3, 2)
sns.lineplot(x=x_line, y=y_line)
plt.xlabel('X-axis')
plt.ylabel('Y-axis') 
plt.title('Line Plot')
plt.show() 



# Show all the plots
#