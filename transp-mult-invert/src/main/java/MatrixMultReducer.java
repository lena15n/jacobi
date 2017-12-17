import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;

public class MatrixMultReducer extends Reducer<Text, Text, Text, Text> {
    private Logger LOG = Logger.getLogger(MatrixMultReducer.class);

    public void reduce(Text key, Iterable<Text> values,
                       Context context) throws IOException, InterruptedException {
        final String hashcode = Integer.toHexString(hashCode());
        LOG.info(String.format("=================== MATRIX TRANSPOSING REDUCER %s", hashcode));

        Configuration conf = context.getConfiguration();
        int columnsA = conf.getInt("columnsA", 0);
        int rowsB = conf.getInt("rowsA", 0);

        double[] rowA = new double[columnsA];
        double[] columnB = new double[rowsB];

        String[] keyData = key.toString().split(" ");
        int i = Integer.valueOf(keyData[0]);
        for (Text value : values) {
            String[] data = value.toString().split(" ");
            if (data[0].equals("A")) {
                rowA[Integer.valueOf(data[1])] = Double.valueOf(data[2]);
            } else {
                columnB[Integer.valueOf(data[1])] = Double.valueOf(data[2]);
            }
        }

        double aat = 0.0;
        for (int j = 0; j < rowA.length; j++) {
            aat += 1.0 * rowA[j] * columnB[j];
        }

        context.write(key, new Text(String.format("%.15f", aat)));
    }
}