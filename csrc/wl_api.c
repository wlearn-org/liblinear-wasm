/*
 * wl_api.c -- C wrapper for LIBLINEAR v2.50
 *
 * Bridges dense JS float64 arrays to LIBLINEAR's sparse feature_node format.
 * Provides buffer-based model I/O via Emscripten MEMFS.
 * Batch prediction (loop in C, not one JS->WASM call per row).
 *
 * Compile with: emcc csrc/wl_api.c upstream/liblinear/linear.cpp ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linear.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- error handling ---------- */

static char last_error[512] = "";

static void set_error(const char *msg) {
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

const char* wl_linear_get_last_error(void) {
  return last_error;
}

/* suppress liblinear's default print to stdout */
static void print_null(const char *s) { (void)s; }

/* ---------- dense-to-sparse conversion ---------- */

/*
 * Convert dense row-major double array to LIBLINEAR sparse format.
 * Pool allocation: one contiguous block of feature_node + row pointer table.
 * Skips zeros. Appends bias feature when bias >= 0.
 * Each row terminated with sentinel { index: -1, value: 0 }.
 *
 * Returns feature_node** (row pointers). Caller must free both
 * the row pointer array and pool[0] (the contiguous node block).
 */
static struct feature_node** dense_to_sparse(
  const double *X, int nrow, int ncol, double bias,
  struct feature_node **pool_out
) {
  /* worst case: ncol nonzeros + optional bias + sentinel per row */
  int max_nodes_per_row = ncol + (bias >= 0 ? 1 : 0) + 1;
  int total_nodes = nrow * max_nodes_per_row;

  struct feature_node *pool = (struct feature_node *)malloc(
    (size_t)total_nodes * sizeof(struct feature_node)
  );
  if (!pool) return NULL;

  struct feature_node **rows = (struct feature_node **)malloc(
    (size_t)nrow * sizeof(struct feature_node *)
  );
  if (!rows) { free(pool); return NULL; }

  struct feature_node *p = pool;
  for (int i = 0; i < nrow; i++) {
    rows[i] = p;
    for (int j = 0; j < ncol; j++) {
      double val = X[i * ncol + j];
      if (val != 0.0) {
        p->index = j + 1;  /* 1-based */
        p->value = val;
        p++;
      }
    }
    if (bias >= 0) {
      p->index = ncol + 1;
      p->value = bias;
      p++;
    }
    p->index = -1;  /* sentinel */
    p->value = 0;
    p++;
  }

  if (pool_out) *pool_out = pool;
  return rows;
}

/* convert single dense row to sparse (reusable scratch buffer) */
static void dense_row_to_sparse(
  const double *row, int ncol, double bias,
  struct feature_node *buf
) {
  struct feature_node *p = buf;
  for (int j = 0; j < ncol; j++) {
    if (row[j] != 0.0) {
      p->index = j + 1;
      p->value = row[j];
      p++;
    }
  }
  if (bias >= 0) {
    p->index = ncol + 1;
    p->value = bias;
    p++;
  }
  p->index = -1;
  p->value = 0;
}

/* ---------- train ---------- */

struct model* wl_linear_train(
  const double *X, int nrow, int ncol,
  const double *y,
  int solver_type, double C, double eps, double bias, double p,
  int nr_weight, const int *weight_label, const double *weight
) {
  set_print_string_function(&print_null);
  last_error[0] = '\0';

  if (!X || !y || nrow <= 0 || ncol <= 0) {
    set_error("Invalid input: X, y, nrow, ncol required");
    return NULL;
  }

  struct feature_node *pool = NULL;
  struct feature_node **sparse_X = dense_to_sparse(X, nrow, ncol, bias, &pool);
  if (!sparse_X) {
    set_error("Failed to allocate sparse matrix");
    return NULL;
  }

  struct problem prob;
  prob.l = nrow;
  prob.n = ncol + (bias >= 0 ? 1 : 0);
  prob.y = (double *)y;
  prob.x = sparse_X;
  prob.bias = bias;

  struct parameter param;
  memset(&param, 0, sizeof(param));
  param.solver_type = solver_type;
  param.C = C > 0 ? C : 1.0;
  param.eps = eps > 0 ? eps : 0.01;
  param.p = p;
  param.nr_weight = nr_weight;
  param.weight_label = (int *)weight_label;
  param.weight = (double *)weight;
  param.regularize_bias = 1;  /* v2.50: default to regularizing bias */

  const char *err = check_parameter(&prob, &param);
  if (err) {
    set_error(err);
    free(pool);
    free(sparse_X);
    return NULL;
  }

  struct model *model = train(&prob, &param);

  free(pool);
  free(sparse_X);

  if (!model) {
    set_error("Training failed");
    return NULL;
  }

  return model;
}

/* ---------- predict (batch) ---------- */

int wl_linear_predict(
  const struct model *m, const double *X, int nrow, int ncol, double *out
) {
  if (!m || !X || !out) {
    set_error("predict: null argument");
    return -1;
  }

  double bias = m->bias;
  /* scratch buffer: ncol + bias + sentinel */
  int buf_size = ncol + (bias >= 0 ? 1 : 0) + 1;
  struct feature_node *buf = (struct feature_node *)malloc(
    (size_t)buf_size * sizeof(struct feature_node)
  );
  if (!buf) {
    set_error("predict: allocation failed");
    return -1;
  }

  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, bias, buf);
    out[i] = predict(m, buf);
  }

  free(buf);
  return 0;
}

int wl_linear_predict_values(
  const struct model *m, const double *X, int nrow, int ncol, double *out
) {
  if (!m || !X || !out) {
    set_error("predict_values: null argument");
    return -1;
  }

  double bias = m->bias;
  int buf_size = ncol + (bias >= 0 ? 1 : 0) + 1;
  struct feature_node *buf = (struct feature_node *)malloc(
    (size_t)buf_size * sizeof(struct feature_node)
  );
  if (!buf) {
    set_error("predict_values: allocation failed");
    return -1;
  }

  int nr_class = m->nr_class;
  /* decision values per row: nr_class for multi-class, 1 for binary */
  int vals_per_row = (nr_class == 2) ? 1 : nr_class;

  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, bias, buf);
    predict_values(m, buf, out + i * vals_per_row);
  }

  free(buf);
  return 0;
}

int wl_linear_predict_probability(
  const struct model *m, const double *X, int nrow, int ncol, double *out
) {
  if (!m || !X || !out) {
    set_error("predict_probability: null argument");
    return -1;
  }

  if (!check_probability_model(m)) {
    set_error("predict_probability: model does not support probability estimates (use LR solver and retrain)");
    return -1;
  }

  double bias = m->bias;
  int buf_size = ncol + (bias >= 0 ? 1 : 0) + 1;
  struct feature_node *buf = (struct feature_node *)malloc(
    (size_t)buf_size * sizeof(struct feature_node)
  );
  if (!buf) {
    set_error("predict_probability: allocation failed");
    return -1;
  }

  int nr_class = m->nr_class;
  for (int i = 0; i < nrow; i++) {
    dense_row_to_sparse(X + i * ncol, ncol, bias, buf);
    predict_probability(m, buf, out + i * nr_class);
  }

  free(buf);
  return 0;
}

/* ---------- model I/O (MEMFS buffer) ---------- */

static int save_counter = 0;

int wl_linear_save_model(
  const struct model *m, char **out_buf, int *out_len
) {
  if (!m || !out_buf || !out_len) {
    set_error("save_model: null argument");
    return -1;
  }

  char path[64];
  snprintf(path, sizeof(path), "/tmp/wl_linear_%d", save_counter++);

  int ret = save_model(path, m);
  if (ret != 0) {
    set_error("save_model: write failed");
    return -1;
  }

  FILE *f = fopen(path, "rb");
  if (!f) {
    set_error("save_model: cannot read back temp file");
    remove(path);
    return -1;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *buf = (char *)malloc((size_t)size);
  if (!buf) {
    fclose(f);
    remove(path);
    set_error("save_model: allocation failed");
    return -1;
  }

  fread(buf, 1, (size_t)size, f);
  fclose(f);
  remove(path);

  *out_buf = buf;
  *out_len = (int)size;
  return 0;
}

struct model* wl_linear_load_model(const char *buf, int len) {
  if (!buf || len <= 0) {
    set_error("load_model: invalid buffer");
    return NULL;
  }

  char path[64];
  snprintf(path, sizeof(path), "/tmp/wl_linear_%d", save_counter++);

  FILE *f = fopen(path, "wb");
  if (!f) {
    set_error("load_model: cannot create temp file");
    return NULL;
  }

  fwrite(buf, 1, (size_t)len, f);
  fclose(f);

  struct model *m = load_model(path);
  remove(path);

  if (!m) {
    set_error("load_model: failed to parse model");
    return NULL;
  }

  return m;
}

/* ---------- buffer management ---------- */

void wl_linear_free_buffer(void *ptr) {
  free(ptr);
}

void wl_linear_free_model(struct model *m) {
  if (m) {
    free_and_destroy_model(&m);
  }
}

/* ---------- model inspection ---------- */

int wl_linear_get_nr_class(const struct model *m) {
  return m ? m->nr_class : 0;
}

int wl_linear_get_nr_feature(const struct model *m) {
  return m ? m->nr_feature : 0;
}

int wl_linear_get_labels(const struct model *m, int *out) {
  if (!m || !out) return -1;
  get_labels(m, out);
  return 0;
}

double wl_linear_get_bias(const struct model *m) {
  return m ? m->bias : -1.0;
}

#ifdef __cplusplus
}
#endif
