// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Apache Lucene 9.12.3 BM25 comparator for hms-bench: same phases and output shape as
// text_worker.py (build, serve, ops). EnglishAnalyzer (Porter stemming, English stop words)
// and the default BM25Similarity (k1 1.2, b 0.75) are Lucene's documented defaults.
//
// Usage: java -cp lucene-core.jar:lucene-analysis-common.jar:classes LuceneBm25
//        <build|serve|ops> <input.json> <store dir> <out.json> <t0 epoch ns> <passes> <ops>
// The input is the {documents, queries} JSON written by hms-bench; it is parsed with a small
// purpose-built reader so the harness needs only the two Lucene jars.

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;

public final class LuceneBm25 {
  static final int K = 100;

  record Doc(String id, String text) {}

  record Q(String id, String text) {}

  /** Minimal JSON reader for the fixed input shape; throws on anything unexpected. */
  static final class Json {
    final String s;
    int p;

    Json(String s) {
      this.s = s;
    }

    void ws() {
      while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++;
    }

    Object value() {
      ws();
      char c = s.charAt(p);
      if (c == '{') {
        p++;
        Map<String, Object> m = new LinkedHashMap<>();
        ws();
        if (s.charAt(p) == '}') {
          p++;
          return m;
        }
        while (true) {
          ws();
          String k = string();
          ws();
          expect(':');
          m.put(k, value());
          ws();
          if (s.charAt(p) == ',') {
            p++;
            continue;
          }
          expect('}');
          return m;
        }
      }
      if (c == '[') {
        p++;
        List<Object> l = new ArrayList<>();
        ws();
        if (s.charAt(p) == ']') {
          p++;
          return l;
        }
        while (true) {
          l.add(value());
          ws();
          if (s.charAt(p) == ',') {
            p++;
            continue;
          }
          expect(']');
          return l;
        }
      }
      if (c == '"') return string();
      int start = p;
      while (p < s.length() && ",]} \n\r\t".indexOf(s.charAt(p)) < 0) p++;
      return s.substring(start, p);
    }

    void expect(char c) {
      if (s.charAt(p) != c) throw new IllegalStateException("expected " + c + " at " + p);
      p++;
    }

    String string() {
      expect('"');
      StringBuilder b = new StringBuilder();
      while (true) {
        char c = s.charAt(p++);
        if (c == '"') return b.toString();
        if (c != '\\') {
          b.append(c);
          continue;
        }
        char e = s.charAt(p++);
        switch (e) {
          case 'n' -> b.append('\n');
          case 't' -> b.append('\t');
          case 'r' -> b.append('\r');
          case 'b' -> b.append('\b');
          case 'f' -> b.append('\f');
          case 'u' -> {
            try {
              b.append((char) Integer.parseInt(s.substring(p, p + 4), 16));
            } catch (NumberFormatException bad) {
              throw new IllegalStateException("bad unicode escape at " + p, bad);
            }
            p += 4;
          }
          default -> b.append(e);
        }
      }
    }
  }

  @SuppressWarnings("unchecked")
  static void load(Path input, List<Doc> docs, List<Q> queries) throws IOException {
    Map<String, Object> root = (Map<String, Object>) new Json(Files.readString(input, StandardCharsets.UTF_8)).value();
    for (Object o : (List<Object>) root.get("documents")) {
      Map<String, Object> m = (Map<String, Object>) o;
      docs.add(new Doc((String) m.get("id"), (String) m.get("text")));
    }
    for (Object o : (List<Object>) root.get("queries")) {
      Map<String, Object> m = (Map<String, Object>) o;
      queries.add(new Q((String) m.get("id"), (String) m.get("text")));
    }
  }

  static List<String> words(String text) {
    Set<String> out = new java.util.TreeSet<>();
    StringBuilder cur = new StringBuilder();
    for (int i = 0; i < text.length(); ) {
      int cp = text.codePointAt(i);
      i += Character.charCount(cp);
      if (Character.isLetterOrDigit(cp)) {
        cur.appendCodePoint(Character.toLowerCase(cp));
      } else if (cur.length() > 0) {
        out.add(cur.toString());
        cur.setLength(0);
      }
    }
    if (cur.length() > 0) out.add(cur.toString());
    return new ArrayList<>(out);
  }

  static Query query(Analyzer analyzer, String text) throws IOException {
    Set<String> terms = new LinkedHashSet<>();
    for (String w : words(text)) {
      try (TokenStream ts = analyzer.tokenStream("body", w)) {
        CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
        ts.reset();
        while (ts.incrementToken()) terms.add(attr.toString());
        ts.end();
      }
    }
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (String t : terms) b.add(new TermQuery(new Term("body", t)), BooleanClause.Occur.SHOULD);
    return b.build();
  }

  static String esc(String s) {
    StringBuilder b = new StringBuilder("\"");
    for (char c : s.toCharArray()) {
      if (c == '"' || c == '\\') b.append('\\').append(c);
      else if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
      else b.append(c);
    }
    return b.append('"').toString();
  }

  static long rssBytes() {
    try {
      Process p = new ProcessBuilder("ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid())).start();
      return Long.parseLong(new String(p.getInputStream().readAllBytes()).trim()) * 1024;
    } catch (Exception e) {
      return 0;
    }
  }

  public static void main(String[] a) throws Exception {
    String phase = a[0];
    Path input = Paths.get(a[1]), store = Paths.get(a[2]), out = Paths.get(a[3]);
    long t0 = Long.parseLong(a[4]);
    int passes = Integer.parseInt(a[5]), opsCount = Integer.parseInt(a[6]);
    List<Doc> docs = new ArrayList<>();
    List<Q> queries = new ArrayList<>();
    load(input, docs, queries);
    Analyzer analyzer = new EnglishAnalyzer();
    StringBuilder json = new StringBuilder("{");
    switch (phase) {
      case "build" -> {
        long start = System.nanoTime();
        try (FSDirectory dir = FSDirectory.open(store);
            IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(analyzer).setSimilarity(new BM25Similarity()))) {
          for (Doc d : docs) w.addDocument(doc(d.id, d.text));
          w.forceMerge(1);
          w.commit();
        }
        json.append("\"documents\":").append(docs.size()).append(",\"ingest_s\":").append((System.nanoTime() - start) / 1e9);
      }
      case "serve" -> {
        long start = System.nanoTime();
        FSDirectory dir = FSDirectory.open(store);
        DirectoryReader reader = DirectoryReader.open(dir);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());
        double openS = (System.nanoTime() - start) / 1e9;
        search(searcher, analyzer, queries.get(0));
        double cold = (System.currentTimeMillis() * 1_000_000L - t0) / 1e9;
        for (int i = 0; i < Math.min(16, queries.size()); i++) search(searcher, analyzer, queries.get(i));
        // JIT warm-up pass: the JVM has no equivalent of the other systems' ahead-of-time code.
        for (Q q : queries) search(searcher, analyzer, q);
        StringBuilder rows = new StringBuilder("[");
        StringBuilder pass = new StringBuilder("[");
        for (int n = 0; n < passes; n++) {
          long[] ns = new long[queries.size()];
          for (int i = 0; i < queries.size(); i++) {
            long t = System.nanoTime();
            String[][] hits = search(searcher, analyzer, queries.get(i));
            ns[i] = System.nanoTime() - t;
            if (n == 0) {
              if (i > 0) rows.append(',');
              rows.append("{\"id\":").append(esc(queries.get(i).id)).append(",\"results\":[");
              for (int h = 0; h < hits.length; h++) {
                if (h > 0) rows.append(',');
                rows.append("{\"documentId\":").append(esc(hits[h][0])).append(",\"lexicalScore\":").append(hits[h][1]).append('}');
              }
              rows.append("]}");
            }
          }
          if (n > 0) pass.append(',');
          pass.append(Arrays.toString(ns));
        }
        json.append("\"open_s\":").append(openS).append(",\"cold_start_s\":").append(cold)
            .append(",\"rows\":").append(rows).append("],\"passes_ns\":").append(pass).append(']');
        reader.close();
        dir.close();
      }
      case "ops" -> {
        try (FSDirectory dir = FSDirectory.open(store);
            IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(analyzer).setSimilarity(new BM25Similarity()))) {
          long t = System.nanoTime();
          for (int i = 0; i < opsCount; i++) {
            Doc d = docs.get(i);
            w.updateDocument(new Term("id", d.id), doc(d.id, d.text + " benchmarkrevision"));
            w.commit();
          }
          double replace = (System.nanoTime() - t) / 1e6 / opsCount;
          t = System.nanoTime();
          for (int i = opsCount; i < 2 * opsCount; i++) {
            w.deleteDocuments(new Term("id", docs.get(i).id));
            w.commit();
          }
          double delete = (System.nanoTime() - t) / 1e6 / opsCount;
          json.append("\"count\":").append(opsCount)
              .append(",\"replace\":{\"supported\":true,\"per_op_ms_durable\":").append(replace)
              .append("},\"delete\":{\"supported\":true,\"per_op_ms_durable\":").append(delete).append('}');
        }
      }
      default -> throw new IllegalArgumentException(phase);
    }
    json.append(",\"peak_rss_bytes_self\":").append(rssBytes()).append('}');
    Files.writeString(out, json.toString());
  }

  static Document doc(String id, String text) {
    Document d = new Document();
    d.add(new StringField("id", id, Field.Store.YES));
    d.add(new TextField("body", text, Field.Store.NO));
    return d;
  }

  static String[][] search(IndexSearcher searcher, Analyzer analyzer, Q q) throws IOException {
    Query query = query(analyzer, q.text);
    ScoreDoc[] hits = searcher.search(query, K).scoreDocs;
    String[][] out = new String[hits.length][];
    StoredFields stored = searcher.storedFields();
    for (int i = 0; i < hits.length; i++) {
      out[i] = new String[] {stored.document(hits[i].doc).get("id"), Float.toString(hits[i].score)};
    }
    return out;
  }
}
