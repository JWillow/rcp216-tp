package org.homework.commons;

import com.google.common.base.Joiner
import com.panayotis.gnuplot.JavaPlot
import com.panayotis.gnuplot.dataset.FileDataSet
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType

class Gnuplot {

    File workDir = new File("/tmp/rcp2016")
    File targetFile = new File("/tmp/rcp2016/gnuplot.txt")

    Encoder<Row> getEncoder() {
        StructType structType = new StructType()
        structType = structType.add("gnuplotFormat", DataTypes.StringType)
        return RowEncoder.apply(structType)
    }

    static void display(Dataset<Row> dataset) {
        Gnuplot plot = new Gnuplot()
        dataset.repartition(1)
                .map(new GnuplotFunction(), plot.getEncoder())
                .write()
                .mode(SaveMode.Overwrite)
                .text(plot.workDir.getPath())
        plot.renameFileTo()
        plot.print()
    }

    private void print() {
        JavaPlot p = new JavaPlot(true)
        p.setTitle("Projection des données spambase")
        p.set("xlabel", "'CP 1'")
        p.set("ylabel", "'CP 2'")
        p.set("zlabel", "'CP 3'")
        p.addPlot(new FileDataSet(targetFile))
        p.plot()
    }


    private renameFileTo() {
        SparkFileUtils.renameFile(workDir, targetFile)
    }

    static class GnuplotFunction implements MapFunction<Row, Row>, Serializable {
        @Override
        Row call(Row row) throws Exception {
            Vector vector = (Vector) row.get(0)
            String value = Joiner.on(" ").skipNulls().join(Arrays.asList(ArrayUtils.toObject(vector.toDense().toArray())))
            return RowFactory.create(value)
        }
    }

}
