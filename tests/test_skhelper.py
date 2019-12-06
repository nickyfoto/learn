from sklearn.datasets import load_iris
iris = load_iris()
print(linear_separability(iris.data, iris.target))