import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA


class IrisPCA:

    def __init__(self, filepath):

        print("Loading iris dataset...")

        self.df = pd.read_csv(filepath)


    def prepare_data(self):

        print("Preparing dataset...")

        X = self.df.drop(columns=["species"])
        y = self.df["species"]

        return X,y


    def apply_pca(self,X,n_components):

        print(f"Applying PCA with {n_components} components...")

        pca = PCA(n_components=n_components)

        components = pca.fit_transform(X)

        return components


    def plot_2d(self,components,labels):

        print("Generating 2D PCA plot...")

        plt.figure()

        plt.scatter(components[:,0],components[:,1])

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.title("PCA with 2 Components")

        plt.savefig("pca_2d.png")

        plt.close()


    def plot_3d(self,components,labels):

        print("Generating 3D PCA plot...")

        fig = plt.figure()

        ax = fig.add_subplot(111,projection='3d')

        ax.scatter(
            components[:,0],
            components[:,1],
            components[:,2]
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        plt.title("PCA with 3 Components")

        plt.savefig("pca_3d.png")

        plt.close()


def run_pca(filepath):

    model = IrisPCA(filepath)

    X,y = model.prepare_data()

    comp2 = model.apply_pca(X,2)
    model.plot_2d(comp2,y)

    comp3 = model.apply_pca(X,3)
    model.plot_3d(comp3,y)

    return True


if __name__ == "__main__":

    run_pca("iris.csv")

    print("\nPCA analysis completed.")
