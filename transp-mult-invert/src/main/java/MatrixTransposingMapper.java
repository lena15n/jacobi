import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.StringTokenizer;

public class MatrixTransposingMapper extends Mapper<Text, Text, IntWritable, Text> {

    private Logger LOG = Logger.getLogger(MatrixTransposingMapper.class);

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        final String hashcode = Integer.toHexString(hashCode());
        LOG.info(String.format("=================== MATRIX TRANSPOSING MAPPER %s", hashcode));

        String rowIdx = key.toString();
        StringTokenizer tokenizer = new StringTokenizer(value.toString());
        int columnIdx = 0;
        while (tokenizer.hasMoreTokens()) {
            context.write(new IntWritable(columnIdx),
                    new Text(String.format("%s %s", rowIdx, tokenizer.nextToken())));
            columnIdx++;
        }
    }
}