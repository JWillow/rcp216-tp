package org.homework.commons

import com.google.common.base.Joiner
import com.google.common.io.Files
import groovy.io.FileType
import org.apache.commons.lang.ArrayUtils
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SaveMode

class SparkFileUtils {

    private static File workDir = new File("/tmp/spark")

    static void renameFile(File directory, File targetFile) {
        directory.eachFile(FileType.FILES) { File file ->
            if (!file.getName().startsWith("part")) {
                return
            }
            Files.move(file, targetFile)
        }
    }

    static void save(Dataset<Row> dataset, File targetFile) {
        dataset.repartition(1)
                .map(new TextFunction(), Encoders.STRING())
                .write()
                .mode(SaveMode.Overwrite)
                .text(workDir.getCanonicalPath())

        renameFile(workDir, targetFile)
    }


    static class TextFunction implements MapFunction<Row, String>, Serializable {
        @Override
        String call(Row row) throws Exception {
            List<Object> toJoin = []
            row.size().times { int i ->
                Object object = row.get(i)
                println "object : $object - ${object.getClass()}"
                if(object instanceof Vector) {
                    Vector vector = (Vector) object
                    toJoin.addAll(Arrays.asList(ArrayUtils.toObject(vector.toDense().toArray())))
                } else {
                    toJoin << object
                }
            }
            return Joiner.on(" ").skipNulls().join(toJoin)
        }
    }
}
