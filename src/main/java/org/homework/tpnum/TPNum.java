package org.homework.tpnum;

import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

public class TPNum {

    public static void main(String[] args) {
        Vector vectDense = Vectors.dense(1.5,0.0,3.5);
        System.out.println(vectDense);

        int dim[] = {0,2};
        double values[] = {1.5,3.5};

        Vector vectCreux1 = Vectors.sparse(3, dim, values);
        System.out.println(vectCreux1.toDense());

        // Calcul de la norme
        System.out.println(Vectors.norm(vectDense, 1));

        // Création d'une matrice dense
        double[] denseMatricContent = {1.2, 3.4, 5.6, 2.3, 4.5, 6.7};
        DenseMatrix denseMatrix = new DenseMatrix(3, 2, denseMatricContent);
        System.out.println(denseMatrix);

    }
}
