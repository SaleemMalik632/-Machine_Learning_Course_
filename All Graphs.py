import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for scatter plot
x_scatter = [1, 2, 3, 4, 5]
y_scatter = [3, 5, 4, 6, 7]

# Sample data for line plot
x_line = [1, 2, 3, 4, 5]
y_line = [3, 5, 4, 6, 7]

# Sample data for bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Sample data for histogram
data_hist = [12, 20, 18, 25, 30, 22, 15, 28, 16, 24]

# Sample data for box plot
data_box = [12, 20, 18, 25, 30, 22, 15, 28, 16, 24]

# Create a scatter plot
plt.subplot(2, 3, 1)
sns.scatterplot(x=x_scatter, y=y_scatter)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')

# Create a line plot
plt.subplot(2, 3, 2)
sns.lineplot(x=x_line, y=y_line)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

# Create a bar plot
plt.subplot(2, 3, 3)
sns.barplot(x=categories, y=values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')

# Create a histogram
plt.subplot(2, 3, 4)
sns.histplot(data_hist)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')

# Create a box plot
plt.subplot(2, 3, 5)
sns.boxplot(data_box)
plt.ylabel('Values')
plt.title('Box Plot')

# Show all the plots
plt.tight_layout()
plt.show()
