
# Extracting data for plotting
data_array = np.array(dataset.select("features").rdd.map(lambda x: x[0].toArray()).collect())
predictions_array = np.array(predictions.select("prediction").rdd.map(lambda x: x[0]).collect())
centers_array = np.array(centers)

# Plotting the clusters
plt.figure(figsize=(8, 6))

# Plot each data point with a different symbol based on cluster
for i in range(len(data_array)):
    if predictions_array[i] == 0:
        plt.scatter(data_array[i][0], data_array[i][1], color='blue', marker='o', label='Cluster 1' if i == 0 else '')
    else:
        plt.scatter(data_array[i][0], data_array[i][1], color='red', marker='x', label='Cluster 2' if i == 0 else '')

# Plot centroids
plt.scatter(centers_array[:, 0], centers_array[:, 1], color='black', marker='*', s=200, label='Centroids')

# Add dummy scatter plots for the legend
plt.scatter([], [], color='blue', marker='o', label='Cluster 1')
plt.scatter([], [], color='red', marker='x', label='Cluster 2')

plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig("/home/ubuntu/output/cluster_plot.png")  # Modify the path as needed

# Close the plot to release resources
plt.close()
