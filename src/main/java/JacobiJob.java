import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.StringTokenizer;

public class JacobiJob {

    public static class JacobiMapper2
            extends Mapper<Text, Text, Text, Text> {

        private Logger LOG = Logger.getLogger(JacobiJob.JacobiMapper2.class);

        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            final String hashcode = Integer.toHexString(hashCode());
            LOG.info(String.format("============= JACOBI MAP %s", hashcode));

            StringTokenizer itr = new StringTokenizer(value.toString());
            int functionType= Integer.valueOf(itr.nextToken());
            int k           = Integer.valueOf(itr.nextToken());
            int n           = Integer.valueOf(itr.nextToken());
            double gamma    = Double.valueOf(itr.nextToken());
            double betta    = Double.valueOf(itr.nextToken());
            double deltaTao = Double.valueOf(itr.nextToken());

            for (int i = 0; i < n; i++) {
                double jacobiResult = 0.0;

                switch (functionType) {
                    case 0: {
                        jacobiResult = derivate(k, betta, deltaTao * i, gamma);
                    }
                    break;
                    case 1: {
                        jacobiResult = p(k, betta, deltaTao * i, gamma);
                    }
                    break;
                    case 2: {
                        jacobiResult = integral(k, betta, deltaTao * i, gamma);
                    }
                    break;
                }
                context.write(new Text(String.format("%d %d", k, i)), new Text(String.format("%.15f %d", jacobiResult, n)));
            }
        }

        private static final double C = 2;

        private static double derivate(int k, double betta, double tao, double gamma) {
            double sum = 0;
            double sign = 1;
            for (int s = 0; s <= k; s++) {
                sum += sign * C(k, s) * doubleC(k + s + betta, s) * pow(2 * s + 1, 1) * Math.exp(-(2 * s + 1) * C * gamma * tao / 2);
                sign *= -1;
            }
            return pow(-C * gamma / 2, 1) * sum;
        }

        private static double p(int k, double betta, double tao, double gamma) {
            double res = 0;
            int sign = 1;
            for (int s = 0; s <= k; s++) {
                res += C(k, s) * doubleC(k + s + betta, s) * sign * Math.exp(-(2 * s + 1) * 2 * gamma * tao / 2);
                sign *= -1;
            }
            return res;
        }

        private static double integral(int k, double betta, double tao, double gamma) {
            int n = 1;
            double res = 0;
            double sign = 1;
            for (int s = 0; s <= k; s++) {
                double binoms = C(k, s) * doubleC(k + s + betta, s);
                double exps = sign * Math.exp(-(2 * s + 1) * C * gamma * tao / 2);
                double rowSum = 0;
                for (int j = 0; j <= n; j++) {
                    rowSum += f(n) * pow(tao, n - j) / (f(n - j) * pow(C * gamma * (2 * s + 1) / 2, j + 1));
                }
                res += binoms * exps * rowSum;
                sign *= -1;
            }
            return res;
        }

        private static double[][] cacheC = new double[100][100];

        static double C(int n, int k) {
            if (cacheC[n][k] != 0) return cacheC[n][k];
            return cacheC[n][k] = f(n) / f(k) / f(n - k);
        }

        private static double f(int n) {
            // or return gamma(n + 1);
            if (n == 0) return 1;
            return f(n - 1) * n;
        }

        private static double pow(double x, int n) {
            if (n == 0) return 1;
            if (n == 1) return x;
            if (n == 2) return x * x;
            return Math.pow(x, n);
        }

        private static double doubleC(double n, double k) {
            return gamma(n + 1) / gamma(k + 1) / gamma(n - k + 1);
        }

        private static double gamma(double x) {
            if (x <= 1e-9) return 1;
    /*
     * Input parameters:
     *       X   -   argument
     * Domain:
     *       0 < X < 171.6
     *    -170 < X < 0, X is not an integer.
     * Relative error:
     * arithmetic   domain     # trials      peak         rms
     * IEEE    -170,-33      20000       2.3e-15     3.3e-16
     * IEEE     -33,  33     20000       9.4e-16     2.2e-16
     * IEEE      33, 171.6   20000       2.3e-15     3.2e-16
     */
            double sgngam, q, z, y, p1, q1;
            int ip, p;
            double[] pp = {1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2, 4.76367800457137231464E-2, 2.07448227648435975150E-1, 4.94214826801497100753E-1, 9.99999999999999996796E-1};
            double[] qq = {-2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3, 1.18139785222060435552E-2, 3.58236398605498653373E-2, -2.34591795718243348568E-1, 7.14304917030273074085E-2, 1.00000000000000000320};
            sgngam = 1;
            q = Math.abs(x);
            if (q > 33.0) {
                if (x < 0.0) {
                    p = (int) Math.floor(q);
                    ip = Math.round(p);
                    if (ip % 2 == 0) {
                        sgngam = -1;
                    }
                    z = q - p;
                    if (z > 0.5) {
                        p = p + 1;
                        z = q - p;
                    }
                    z = q * Math.sin(Math.PI * z);
                    z = Math.abs(z);
                    z = Math.PI / (z * gammastirf(q));
                } else {
                    z = gammastirf(x);
                }
                y = sgngam * z;
                return y;
            }
            z = 1;
            while (x >= 3) {
                x = x - 1;
                z = z * x;
            }
            while (x < 0) {
                if (x > -0.000000001) {
                    y = z / ((1 + 0.5772156649015329 * x) * x);
                    return y;
                }
                z = z / x;
                x = x + 1;
            }
            while (x < 2) {
                if (x < 0.000000001) {
                    y = z / ((1 + 0.5772156649015329 * x) * x);
                    return y;
                }
                z = z / x;
                x = x + 1.0;
            }
            if (x == 2) {
                y = z;
                return y;
            }
            x = x - 2.0;
            p1 = pp[0];
            for (int i = 1; i < 7; i++) {
                p1 = pp[i] + p1 * x;
            }
            q1 = qq[0];
            for (int i = 1; i < 8; i++) {
                q1 = qq[i] + q1 * x;
            }
            return z * p1 / q1;
        }

        private static double gammastirf(double x) {
            double p1, w, y, v;
            w = 1 / x;
            double[] pp = {7.87311395793093628397E-4, -2.29549961613378126380E-4, -2.68132617805781232825E-3, 3.47222221605458667310E-3, 8.33333333333482257126E-2};
            p1 = pp[0];
            for (int i = 1; i < 5; i++) {
                p1 = pp[i] + p1 * x;
            }
            w = 1 + w * p1;
            y = Math.exp(x);
            if (x > 143.01608) {
                v = Math.pow(x, 0.5 * x - 0.25);
                y = v * (v / y);
            } else {
                y = Math.pow(x, x - 0.5) / y;
            }
            return 2.50662827463100050242 * y * w;
        }
    }

    public static class JacobiReducer
            extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values,
                           Context context) throws IOException, InterruptedException {}
    }

    /**
    * @param args <br>
    *       args[0] Path to input file in local file system<br>
    *       args[1] Output file name (without extension pls)
    *
    * */
    public static void main(String[] args) throws Exception {
        final Log LOG = LogFactory.getLog(JacobiJob.class);

        String localInputFile = args[0];

        Configuration conf = new Configuration();
        FileSystem hdfs = FileSystem.get(conf);
        Path outputPath = prepareOutputFolder(hdfs, args[1]);
        Job job = Job.getInstance(conf, "Jacobi job");
        job.setJarByClass(JacobiJob.class);
        job.setReducerClass(JacobiReducer.class);
        job.setNumReduceTasks(0);
        FileOutputFormat.setOutputPath(job, outputPath);

        //TODO: hack
        String inputFolderName = "jacobi-in";
        if (hdfs.exists(new Path(inputFolderName))) {
            hdfs.delete(new Path(inputFolderName), true);
        }

        Scanner sc = new Scanner(new File(localInputFile));
        int i = 0;
        while (sc.hasNext()) {
            Path inputPath = new Path(inputFolderName + "/" + localInputFile + i);
            FSDataOutputStream fos = hdfs.create(inputPath);
            fos.getWrappedStream().write(sc.nextLine().getBytes());
            fos.close();

            MultipleInputs.addInputPath(job, inputPath, KeyValueTextInputFormat.class, JacobiMapper2.class);

            i++;
        }
        sc.close();
        //TODO: hack

        if (job.waitForCompletion(true)) {
            // Merge part-r-00000 files after Reduce phase into one file,
            // copy to local file system and
            // remove output folder in hdfs
            FileUtil.copyMerge(hdfs, outputPath, FileSystem.getLocal(conf),
                    new Path(outputPath.getName()), true,
                    conf, null);
        }
    }

    private static Path prepareOutputFolder(FileSystem hdfs, String output) throws IOException {
        Path outputPath = new Path(output);

        if (hdfs.exists(outputPath)) {
            hdfs.delete(outputPath, true);
        }

        return outputPath;
    }
}