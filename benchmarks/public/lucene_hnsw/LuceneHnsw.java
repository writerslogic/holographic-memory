// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Lucene HNSW through its own graph API (no index files, no PyLucene): builds an on-heap HNSW
// graph over train.f32 and times single-threaded, one-query-at-a-time searches over test.f32,
// as evaluate.py does for the other systems. Prints one JSON object on stdout and writes the
// first repeat's top-k ids per ef to <outDir>/lucene_M<m>_ef<ef>.i32 (int32 little-endian,
// nQueries x k) for evaluate.py to score.
//
// Usage: java -cp lucene-core.jar:classes LuceneHnsw <train.f32> <test.f32> <dim> <M>
//        <beamWidth> <ef,ef,...> <k> <repeats> <outDir> <nQueries|0> <angular|euclidean>
//        <maxLoad> <maxWaitSecs> <buildThreads>

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.hnsw.HnswConcurrentMergeBuilder;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

public final class LuceneHnsw {
  /** In-memory vectors; every ordinal owns its array, so sharing one instance is safe. */
  static final class Vectors implements RandomAccessVectorValues.Floats {
    final float[][] rows;
    final int dim;

    Vectors(float[][] rows, int dim) {
      this.rows = rows;
      this.dim = dim;
    }

    @Override
    public int size() {
      return rows.length;
    }

    @Override
    public int dimension() {
      return dim;
    }

    @Override
    public float[] vectorValue(int ord) {
      return rows[ord];
    }

    @Override
    public Vectors copy() {
      return this;
    }
  }

  static float[][] read(Path path, int dim, boolean normalize) throws IOException {
    try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
      int n = (int) (ch.size() / (4L * dim));
      float[][] out = new float[n][];
      int perChunk = Math.max(1, (1 << 22) / dim);
      ByteBuffer buf = ByteBuffer.allocateDirect(perChunk * dim * 4).order(ByteOrder.LITTLE_ENDIAN);
      int at = 0;
      while (at < n) {
        int rows = Math.min(perChunk, n - at);
        buf.clear();
        buf.limit(rows * dim * 4);
        while (buf.hasRemaining()) {
          if (ch.read(buf) < 0) {
            throw new IOException("short read in " + path);
          }
        }
        buf.flip();
        for (int i = 0; i < rows; i++) {
          float[] v = new float[dim];
          buf.asFloatBuffer().get(v);
          buf.position(buf.position() + dim * 4);
          if (normalize) {
            double s = 0;
            for (float x : v) {
              s += (double) x * x;
            }
            if (s > 0) {
              float inv = (float) (1 / Math.sqrt(s));
              for (int j = 0; j < dim; j++) {
                v[j] *= inv;
              }
            }
          }
          out[at + i] = v;
        }
        at += rows;
      }
      return out;
    }
  }

  /** The same load gate as evaluate.py: wait for the 1-minute load, within one total budget. */
  static double[] waitForIdle(double maxLoad, long deadlineNanos) throws InterruptedException {
    while (true) {
      double load = ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage();
      if (load < maxLoad) {
        return new double[] {load, 1};
      }
      if (System.nanoTime() >= deadlineNanos) {
        return new double[] {load, 0};
      }
      Thread.sleep(5000);
    }
  }

  public static void main(String[] a) throws Exception {
    Path train = Paths.get(a[0]);
    Path test = Paths.get(a[1]);
    int dim = Integer.parseInt(a[2]);
    int m = Integer.parseInt(a[3]);
    int beam = Integer.parseInt(a[4]);
    String[] efs = a[5].split(",");
    int k = Integer.parseInt(a[6]);
    int repeats = Integer.parseInt(a[7]);
    Path outDir = Paths.get(a[8]);
    int nq = Integer.parseInt(a[9]);
    boolean angular = a[10].equals("angular");
    double maxLoad = Double.parseDouble(a[11]);
    long maxWaitSecs = Long.parseLong(a[12]);
    int threads = Integer.parseInt(a[13]);
    Files.createDirectories(outDir);

    float[][] tr = read(train, dim, angular);
    float[][] te = read(test, dim, angular);
    if (nq > 0 && nq < te.length) {
      te = Arrays.copyOf(te, nq);
    }
    Vectors v = new Vectors(tr, dim);
    VectorSimilarityFunction sim =
        angular ? VectorSimilarityFunction.DOT_PRODUCT : VectorSimilarityFunction.EUCLIDEAN;
    RandomVectorScorerSupplier supplier =
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(sim, v);

    long t0 = System.nanoTime();
    OnHeapHnswGraph graph;
    String builder;
    if (threads > 1) {
      ExecutorService pool = Executors.newFixedThreadPool(threads);
      try {
        // The graph object comes from a single-threaded builder (its constructor is not public).
        OnHeapHnswGraph empty = HnswGraphBuilder.create(supplier, m, beam, 42L, v.size()).getGraph();
        graph =
            new HnswConcurrentMergeBuilder(new TaskExecutor(pool), threads, supplier, m, beam, empty, null)
                .build(v.size());
      } finally {
        pool.shutdown();
      }
      builder = "HnswConcurrentMergeBuilder(" + threads + " workers)";
    } else {
      graph = HnswGraphBuilder.create(supplier, m, beam, 42L).build(v.size());
      builder = "HnswGraphBuilder";
    }
    double buildSecs = (System.nanoTime() - t0) / 1e9;
    long graphBytes = graph.ramBytesUsed();
    long indexBytes = graphBytes + (long) v.size() * dim * 4;

    long deadline = System.nanoTime() + maxWaitSecs * 1_000_000_000L;
    StringBuilder sb = new StringBuilder();
    sb.append("{\"builder\":\"").append(builder).append("\",\"build_secs\":").append(buildSecs);
    sb.append(",\"index_bytes\":").append(indexBytes).append(",\"graph_ram_bytes\":").append(graphBytes);
    sb.append(",\"n_queries\":").append(te.length).append(",\"sweep\":[");
    int[] ids = new int[te.length * k];
    for (int ei = 0; ei < efs.length; ei++) {
      int ef = Math.max(Integer.parseInt(efs[ei].trim()), k);
      double[] qps = new double[repeats];
      double[] loads = new double[repeats];
      boolean gate = true;
      for (int r = 0; r < repeats; r++) {
        double[] g = waitForIdle(maxLoad, deadline);
        loads[r] = g[0];
        gate &= g[1] > 0;
        long t = System.nanoTime();
        for (int i = 0; i < te.length; i++) {
          RandomVectorScorer scorer = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(sim, v, te[i]);
          TopKnnCollector c = new TopKnnCollector(ef, Integer.MAX_VALUE);
          HnswGraphSearcher.search(scorer, c, graph, null);
          if (r == 0) {
            ScoreDoc[] sd = c.topDocs().scoreDocs;
            for (int j = 0; j < k; j++) {
              ids[i * k + j] = j < sd.length ? sd[j].doc : -1;
            }
          }
        }
        qps[r] = te.length / ((System.nanoTime() - t) / 1e9);
      }
      Path f = outDir.resolve("lucene_M" + m + "_ef" + ef + ".i32");
      ByteBuffer out = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
      out.asIntBuffer().put(ids);
      Files.write(f, out.array());
      sb.append(ei > 0 ? "," : "").append("{\"ef\":").append(ef).append(",\"qps_runs\":").append(Arrays.toString(qps));
      sb.append(",\"load_1m_before_runs\":").append(Arrays.toString(loads)).append(",\"load_gate_met\":").append(gate);
      sb.append(",\"ids_file\":\"").append(f.toString().replace("\\", "\\\\")).append("\"}");
      System.err.println("ef=" + ef + " qps=" + Arrays.toString(qps));
    }
    sb.append("]}");
    System.out.println(sb);
  }
}
