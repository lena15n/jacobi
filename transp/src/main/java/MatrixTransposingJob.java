import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.StringTokenizer;

public class MatrixTransposingJob {

    public static class MatrixTransposingMapper
            extends Mapper<Text, Text, IntWritable, Text> {

        private Logger LOG = Logger.getLogger(MatrixTransposingJob.MatrixTransposingMapper.class);

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

    public static class MatrixTransposingReducer
            extends Reducer<IntWritable, Text, IntWritable, Text> {
        private Logger LOG = Logger.getLogger(MatrixTransposingJob.MatrixTransposingReducer.class);

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

            context.write(key, new Text(sb.toString()));
        }
    }

    /**
     * @param args <br>
     *       args[0] Path to input file in local file system<br>
     *       args[1] Output file name (without extension pls)
     *
     * */
    public static void main(String[] args) throws Exception {
        final Log LOG = LogFactory.getLog(MatrixTransposingJob.class);

        String localInputFile = args[0];

        Configuration conf = new Configuration();
        conf.setInt("rowsCount", Integer.valueOf(args[2]));
        FileSystem hdfs = FileSystem.get(conf);
        Path inputPath = putInputFileInHdfs(hdfs, localInputFile);
        Path outputPath = prepareOutputFolder(hdfs, args[1]);

        Job job = Job.getInstance(conf, "Matrix Transposing");
        job.setJarByClass(MatrixTransposingJob.class);
        job.setMapperClass(MatrixTransposingMapper.class);
        job.setReducerClass(MatrixTransposingReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class); // key as Text, not LongWritable
        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);

        if (job.waitForCompletion(true)) {
            // Merge part-r-00000 files after Reduce phase into one file,
            // copy to local file system and
            // remove output folder in hdfs
            FileUtil.copyMerge(hdfs, outputPath, FileSystem.getLocal(conf),
                    new Path(outputPath.getName()), true,
                    conf, null);
        }
    }

    private static Path putInputFileInHdfs(FileSystem hdfs, String file) throws IOException {
        Path inputPath = new Path("transpose-in/" + file);
        hdfs.copyFromLocalFile(false, true, new Path(file), inputPath);

        return inputPath;
    }

    private static Path prepareOutputFolder(FileSystem hdfs, String output) throws IOException {
        Path outputPath = new Path(output);

        if (hdfs.exists(outputPath)) {
            hdfs.delete(outputPath, true);
        }

        return outputPath;
    }
}