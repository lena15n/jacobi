import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.log4j.Logger;

import java.io.IOException;

public class MatrixTransposingReducer extends Reducer<IntWritable, Text, Text, Text> {
    private Logger LOG = Logger.getLogger(MatrixTransposingReducer.class);

    public void reduce(IntWritable key, Iterable<Text> values,
                       Context context) throws IOException, InterruptedException {
        final String hashcode = Integer.toHexString(hashCode());
        LOG.info(String.format("=================== MATRIX TRANSPOSING REDUCER %s", hashcode));

        StringBuilder sb = new StringBuilder();
        int rows = context.getConfiguration().getInt("rowsCount", 0);
        String[] vector = new String[rows];

        for (Text value : values) {
            String[] rowIdxAndElem = value.toString().split(" ");
            vector[Integer.valueOf(rowIdxAndElem[0])] = rowIdxAndElem[1];
        }

        for (int i = 0; i < rows; i++) {
            sb.append(vector[i]).append(" ");
        }

        context.write(new Text(key.toString()), new Text(sb.toString()));
    }
}