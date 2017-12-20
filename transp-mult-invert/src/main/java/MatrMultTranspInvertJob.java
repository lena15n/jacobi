import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MapFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.security.Key;
import java.util.Scanner;
import java.util.StringTokenizer;

public class MatrMultTranspInvertJob {
    /**
     * @param args <br>
     *             args[0] Path to input file in local file system<br>
     *             args[1] Output file name (without extension pls)
     */
    public static void main(String[] args) throws Exception {
        Configuration transpConf = new Configuration();
        transpConf.setInt("rowsCount", Integer.valueOf(args[2]));

        String localInputFile = args[0];
        String tranInFolder = "matrix-in/";
        String tranOutFolder = "matrix-out";
        FileSystem hdfs = FileSystem.get(transpConf);
        Path inputPath = putInputFileInHdfs(hdfs, tranInFolder, localInputFile);
        Path transpOutputPath = prepareOutputFolder(hdfs, tranOutFolder);

        Job transpJob = Job.getInstance(transpConf, "Part 1: Matrix Transposing");
        transpJob.setJarByClass(MatrMultTranspInvertJob.class);
        transpJob.setMapperClass(MatrixTransposingMapper.class);
        transpJob.setReducerClass(MatrixTransposingReducer.class);
        transpJob.setMapOutputKeyClass(IntWritable.class);
        transpJob.setMapOutputValueClass(Text.class);
        transpJob.setInputFormatClass(KeyValueTextInputFormat.class);
        FileInputFormat.addInputPath(transpJob, inputPath);
        FileOutputFormat.setOutputPath(transpJob, transpOutputPath);

        if (transpJob.waitForCompletion(true)) {
            Configuration multConf = new Configuration();
            multConf.setInt("rowsA", Integer.valueOf(args[2]));
            multConf.setInt("columnsA", Integer.valueOf(args[3]));
            Job multJob = Job.getInstance(multConf, "Part 2: Matrix Multiplication");
            multJob.setJarByClass(MatrMultTranspInvertJob.class);
            MultipleInputs.addInputPath(multJob, inputPath, KeyValueTextInputFormat.class, MatrixMultMapperA.class);
            MultipleInputs.addInputPath(multJob, transpOutputPath, KeyValueTextInputFormat.class, MatrixMultMapperB.class);
            Path outputPath = prepareOutputFolder(hdfs, args[1]);
            multJob.setMapOutputKeyClass(Text.class);
            multJob.setMapOutputValueClass(Text.class);
            FileOutputFormat.setOutputPath(multJob, outputPath);
            multJob.setReducerClass(MatrixMultReducer.class);
            if (multJob.waitForCompletion(true)) {
                hdfs.delete(inputPath, true);
                hdfs.delete(transpOutputPath, true);

                // Merge part-r-00000 files after Reduce phase into one file,
                // copy to local file system and
                // remove output folder in hdfs
                FileUtil.copyMerge(hdfs, outputPath, FileSystem.getLocal(transpConf),
                        new Path(outputPath.getName()), true,
                        transpConf, null);
            }
        }
    }

    private static Path putInputFileInHdfs(FileSystem hdfs, String hdfsInputFolder, String file) throws IOException {
        Path inputPath = new Path(hdfsInputFolder + "/" + file);
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