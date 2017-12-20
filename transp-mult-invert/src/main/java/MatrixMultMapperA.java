import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.StringTokenizer;

public class MatrixMultMapperA extends Mapper<Text, Text, Text, Text> {

    private Logger LOG = Logger.getLogger(MatrixMultMapperA.class);

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        final String hashcode = Integer.toHexString(hashCode());
        LOG.info(String.format("=================== MATRIX MAPPER A %s", hashcode));

        // rowIdx   a[i, j] a[i, j+1] ... a[i, columnsA]
        Configuration conf = context.getConfiguration();
        int rows = conf.getInt("rowsA", 0);
        int columnsA = conf.getInt("columnsA", 0);
        StringTokenizer tokenizer = new StringTokenizer(value.toString());
        LOG.info(String.format("=================== MATRIX MAPPER A %s: key: %s, value: %s", hashcode,
                key.toString(), value.toString()));

        String i = key.toString();
        int columnsB = rows;
        for (int j = 0; j < columnsA; j++) {
            String a = tokenizer.nextToken();
            for (int k = 0; k < columnsB; k++) {
                // Produces (i, k) (A, j, aij)
                context.write(new Text(String.format("%s %d", i, k)),
                        new Text(String.format("%s %d %s", "A", j, a)));
            }
        }
    }
}