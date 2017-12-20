import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.StringTokenizer;

public class MatrixMultMapperB extends Mapper<Text, Text, Text, Text> {

    private Logger LOG = Logger.getLogger(MatrixMultMapperB.class);

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        final String hashcode = Integer.toHexString(hashCode());
        LOG.info(String.format("=================== MATRIX MAPPER B %s", hashcode));

        // rowIdx   b[j, k] b[j, k+1] ... b[j, rowsA] // B = At
        Configuration conf = context.getConfiguration();
        int rowsA = conf.getInt("rowsA", 0);
        int columnsB = conf.getInt("rowsA", 0);
        StringTokenizer tokenizer = new StringTokenizer(value.toString());
        LOG.info(String.format("=================== MATRIX MAPPER B %s: key: %s, value: %s", hashcode,
                key.toString(), value.toString()));
        String j = key.toString();


        for (int k = 0; k < columnsB; k++) {
            String at = tokenizer.nextToken();
            for (int i = 0; i < rowsA; i++) {
                // Produces (i, k) (B, j, bjk)
                context.write(new Text(String.format("%s %d", i, k)),
                        new Text(String.format("%s %s %s", "B", j, at)));
            }
        }
    }
}