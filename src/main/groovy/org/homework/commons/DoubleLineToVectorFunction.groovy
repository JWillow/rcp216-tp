package org.homework.commons

import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row

class DoubleLineToVectorFunction implements MapFunction<Row, Vector> {

    @Override
    Vector call(Row row) throws Exception {
        double[] values = new double[row.size()]
        row.size().times {int i ->
            values[i] = Double.parseDouble((String) row.get(i))
        }
        return Vectors.dense(values)
    }
}
