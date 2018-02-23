"""
Lifted from: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#3-a-computing-the-scatter-matrix
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


"""
TABLE OF CONTENTS
Line 29: Make sample data
Line 44: Plot sample data
Line 73: Matrix manipulation
Line 93: Compute eigen-stuffs
Line 115: Plot eigenstuffs
Line 146: Make W, our matrix defining the transformation we'd like to apply and transform
Line 172: Plot the reduced data
"""





### 1: Make some sample data

mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

mu_vec2 = np.array([10, 10, 10])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T

# Since, for PCA we don't actually know the groups, let's just treat the two as one set.
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
print "Sample data made"




### 1.1: Make a test plot to get a feel for where the data are
plot_1p1 = False
if plot_1p1:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10

    ax.plot(class1_sample[0, :],
            class1_sample[1, :],
            class1_sample[2, :],
            'o', markersize=8,
            color='blue', alpha=0.5,
            label='class1')
    ax.plot(class2_sample[0, :],
            class2_sample[1, :],
            class2_sample[2, :],
            '^', markersize=8,
            alpha=0.5, color='red',
            label='class2')

    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    plt.show(block=False)




### 2: Begin matrix manipulation

# Compute the mean vector for the total data set. We will use this to compute the covariance matrix
mean_x = np.mean(all_samples[0, :])
mean_y = np.mean(all_samples[1, :])
mean_z = np.mean(all_samples[2, :])
mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
# Print it out if you want
print 'Mean Vector: \n', mean_vector


# Choose whether you'd like to use a scatter matrix or covariance matrix
# Covariance matrix is scaled by 1/(N-1)

scatter_matrix = np.zeros((3, 3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:, i].reshape(3, 1) - mean_vector).dot((all_samples[:, i].reshape(3, 1) - mean_vector).T)

cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])



### Compute eigen-stuffs to inform us on which dimensions we can drop

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

# Want to print out info about the eigen-stuffs?
print_info = True

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:, i].reshape(1, 3).T
    eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T

    if print_info:
        print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
        print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
        print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
        print(40 * '-')


# Just to get a feel for what these eigen-stuffs mean, let's plot them:
plot_eigen_stuffs = False
if plot_eigen_stuffs:
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=9, color='red', alpha=.5)
    for v in eig_vec_sc.T:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')
    plt.show(block=False)



"""
What we want now is to sort the eigen-vectors by their eigenvalues, which serve as a proxy for variance along that dimension, and then keep only the k longest ones (cut off the bottom N-k). We can then transform our data onto the new d*k subspace using our k favorite eigenvectors.
"""


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])


# Create a d*k dimensional eigenvector matrix W to use to transform our data
matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print('Matrix W:\n', matrix_w)

# Finally, make the transformation using the equation:
# y = (W.T)(x)
transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2, 40), "The matrix is not 2x40 dimensional."


# Plot the resulting reduced data set
final_plot = True
if final_plot:
    plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show(block=False)




















# The End
