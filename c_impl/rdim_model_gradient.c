#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_GUI
#include "viewer.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DT 0.001
#define SNAPSHOT_STRIDE_DEFAULT 1000
#define OUTPUT_DIR_DEFAULT "c_rdim_outputs"

static const double PIXEL_SIZE_UM = 1.0;
static int GRID_N = 256;
static double DOMAIN_SIZE_UM = 256.0;
static double DH = 1.0;
static double DH2_INV = 1.0;
static size_t GRID_TOTAL = (size_t)256 * (size_t)256;

static size_t num_c6_concs(void);
static const double C6_CONCS[];

static void set_grid_geometry(int n)
{
    if (n < 4)
    {
        n = 4;
    }
    GRID_N = n;
    DOMAIN_SIZE_UM = (double)GRID_N * PIXEL_SIZE_UM;
    DH = DOMAIN_SIZE_UM / (double)GRID_N;
    DH2_INV = 1.0 / (DH * DH);
    GRID_TOTAL = (size_t)GRID_N * (size_t)GRID_N;
}

/* Build 16 starting C6e values:
 * - 3 below 0.9×switch (0.01, 0.05, 0.25 factors)
 * - 10 evenly spaced between 0.9× and 1.1× switch
 * - 3 above (1.2, 1.5, 2.0 factors)
 */
static size_t build_c6_grid(size_t dm_idx, double **out_grid)
{
    if (dm_idx >= num_c6_concs())
    {
        return 0;
    }
    const double base = C6_CONCS[dm_idx];
    double factors[16] = {
        0.01, 0.05, 0.25,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.2, 1.5, 2.0};
    for (int i = 0; i < 10; ++i)
    {
        factors[3 + i] = 0.9 + (0.2 * (double)i) / 9.0;
    }
    double *grid = (double *)malloc(sizeof(double) * 16);
    if (!grid)
    {
        return 0;
    }
    for (int i = 0; i < 16; ++i)
    {
        grid[i] = factors[i] * base;
    }
    *out_grid = grid;
    return 16;
}

/* Biochemical parameters */
static const double k_tx = 1e-1;
static const double k_tr = 5e0;
static const double d_mrna = 5e-1;
static const double d_p = 5e-2;

static const double K_1R = 1.0 / 0.1;
static const double K_1S = 1.0 / 0.1;
static const double K_2R = 1.0 / 0.02;
static const double K_2S = 1.0 / 0.02;
static const double K_GR = 1.0 / 0.10;
static const double K_GS = 1.0 / 0.10;
static const double KM_ENZ = 10.0;

static const double k_LUXI = 0.04;
static const double k_LASI = 0.04;

static const double d_HSLi = 0.01;
static const double d_HSLe = 0.001;

static const double K_TET = 1.0 / 0.001;
static const double K_LAC = 1.0 / 0.100;
static const double n_R = 2.0;
static const double n_S = 2.0;
static const double n_REP1 = 2.0;
static const double n_REP2 = 4.0;
static const double b_0 = 5e-5;
static const double b_1 = 2.5e-4;

static const double alpha_const = (k_tx * k_tr) / (d_mrna * d_p);
static const double c_const = alpha_const * alpha_const;
static const double ktr_over_dmrna = k_tr / d_mrna;

#ifdef WITH_GUI
typedef struct
{
    double min;
    double max;
    double mean;
} FieldStats;

static FieldStats compute_field_stats(const double *field)
{
    FieldStats stats = {field[0], field[0], 0.0};
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double value = field[k];
        stats.mean += value;
        if (value < stats.min)
        {
            stats.min = value;
        }
        if (value > stats.max)
        {
            stats.max = value;
        }
    }
    stats.mean /= (double)GRID_TOTAL;
    return stats;
}

static inline uint8_t encode_uint8(double value, double min_val, double max_val)
{
    const double denom = max_val - min_val;
    if (denom <= 1e-12)
    {
        return 0;
    }
    double scaled = (value - min_val) / denom;
    if (scaled < 0.0)
    {
        scaled = 0.0;
    }
    else if (scaled > 1.0)
    {
        scaled = 1.0;
    }
    const int sample = (int)lrint(scaled * 255.0);
    return (uint8_t)((sample < 0) ? 0 : (sample > 255 ? 255 : sample));
}

static double compute_magnetization(const double *gfp, const double *rfp)
{
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double g = gfp[k];
        const double r = rfp[k];
        numerator += r - g;
        denominator += r + g;
    }
    if (denominator < 1e-12)
    {
        return 0.0;
    }
    return numerator / denominator;
}

static void compose_rgb_frame(const double *gfp, const double *rfp, uint8_t *rgb,
                              FieldStats *out_gfp, FieldStats *out_rfp, double *out_mag)
{
    FieldStats g_stats = compute_field_stats(gfp);
    FieldStats r_stats = compute_field_stats(rfp);
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        rgb[3 * k + 0] = encode_uint8(rfp[k], r_stats.min, r_stats.max);
        rgb[3 * k + 1] = encode_uint8(gfp[k], g_stats.min, g_stats.max);
        rgb[3 * k + 2] = 0;
    }
    if (out_mag)
    {
        *out_mag = compute_magnetization(gfp, rfp);
    }
    if (out_gfp)
    {
        *out_gfp = g_stats;
    }
    if (out_rfp)
    {
        *out_rfp = r_stats;
    }
}
#endif

static inline size_t idx(int i, int j)
{
    return (size_t)i * (size_t)GRID_N + (size_t)j;
}

/* -------------------------------------------------------------------------- */
/* RNG utilities (xorshift128+)                                               */
/* -------------------------------------------------------------------------- */
typedef struct
{
    uint64_t s[2];
} RngState;

static inline uint64_t splitmix64_next(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(RngState *rng, uint64_t seed)
{
    uint64_t x = seed ? seed : 0x1234567890ABCDEFULL;
    rng->s[0] = splitmix64_next(&x);
    rng->s[1] = splitmix64_next(&x);
}

static inline uint64_t rng_next(RngState *rng)
{
    uint64_t s1 = rng->s[0];
    const uint64_t s0 = rng->s[1];
    rng->s[0] = s0;
    s1 ^= s1 << 23;
    rng->s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return rng->s[1] + s0;
}

static inline double rng_uniform(RngState *rng)
{
    const uint64_t value = rng_next(rng);
    return (value >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_normal(RngState *rng, double mean, double stddev)
{
    double u1 = rng_uniform(rng);
    double u2 = rng_uniform(rng);
    u1 = fmax(u1, 1e-12);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 2.0 * M_PI * u2;
    return mean + stddev * radius * cos(angle);
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */
static inline double hill_power(double value, double exponent)
{
    if (exponent == 2.0)
    {
        return value * value;
    }
    if (exponent == 4.0)
    {
        double sq = value * value;
        return sq * sq;
    }
    return pow(value, exponent);
}

static inline double promoter_function(double HSL, double REP, double K_GENE_HSL,
                                       double K_DIMER, double K_TF_HSL,
                                       double K_REP, double n_HSL, double n_REP)
{
    const double tf_bind = K_TF_HSL * HSL;
    const double frac = tf_bind / (1.0 + tf_bind);
    const double activator = K_GENE_HSL * K_DIMER * c_const * hill_power(frac, n_HSL);
    const double rep_term = hill_power(K_REP * REP, n_REP);
    const double denom = 1.0 + activator + rep_term;
    return activator / denom;
}

static inline double clip_nonnegative(double value)
{
    return value < 0.0 ? 0.0 : value;
}

static double array_mean(const double *arr, size_t count)
{
    double sum = 0.0;
    for (size_t k = 0; k < count; ++k)
    {
        sum += arr[k];
    }
    return sum / (double)count;
}

static double compute_magnetization_stats(const double *gfp, const double *rfp)
{
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        numerator += rfp[k] - gfp[k];
        denominator += rfp[k] + gfp[k];
    }
    if (denominator < 1e-12)
    {
        return 0.0;
    }
    return numerator / denominator;
}

static int ensure_dir(const char *path)
{
    if (mkdir(path, 0755) == -1)
    {
        if (errno == EEXIST)
        {
            return 0;
        }
        return -1;
    }
    return 0;
}

static void swap_arrays(double **a, double **b)
{
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

/* -------------------------------------------------------------------------- */
/* Data containers                                                            */
/* -------------------------------------------------------------------------- */
typedef struct
{
    double *LUXI;
    double *LASI;
    double *C6i;
    double *C12i;
    double *C6e;
    double *C12e;
    double *TETR;
    double *LACI;
    double *RFP;
    double *GFP;
    double *LUXI_next;
    double *LASI_next;
    double *C6i_next;
    double *C12i_next;
    double *C6e_next;
    double *C12e_next;
    double *TETR_next;
    double *LACI_next;
    double *RFP_next;
    double *GFP_next;
    double *lap_C6;
    double *lap_C12;
    double *pLT;
    double *pLL;
#ifdef WITH_GUI
    double *mag_history;
#endif
} SimulationArrays;

static int allocate_arrays(SimulationArrays *arr)
{
    memset(arr, 0, sizeof(*arr));
#define ALLOC_FIELD(name)                               \
    do                                                  \
    {                                                   \
        arr->name = calloc(GRID_TOTAL, sizeof(double)); \
        if (!arr->name)                                 \
        {                                               \
            return -1;                                  \
        }                                               \
    } while (0)

    ALLOC_FIELD(LUXI);
    ALLOC_FIELD(LASI);
    ALLOC_FIELD(C6i);
    ALLOC_FIELD(C12i);
    ALLOC_FIELD(C6e);
    ALLOC_FIELD(C12e);
    ALLOC_FIELD(TETR);
    ALLOC_FIELD(LACI);
    ALLOC_FIELD(RFP);
    ALLOC_FIELD(GFP);
    ALLOC_FIELD(LUXI_next);
    ALLOC_FIELD(LASI_next);
    ALLOC_FIELD(C6i_next);
    ALLOC_FIELD(C12i_next);
    ALLOC_FIELD(C6e_next);
    ALLOC_FIELD(C12e_next);
    ALLOC_FIELD(TETR_next);
    ALLOC_FIELD(LACI_next);
    ALLOC_FIELD(RFP_next);
    ALLOC_FIELD(GFP_next);
    ALLOC_FIELD(lap_C6);
    ALLOC_FIELD(lap_C12);
    ALLOC_FIELD(pLT);
    ALLOC_FIELD(pLL);
#ifdef WITH_GUI
    arr->mag_history = NULL;
#endif

#undef ALLOC_FIELD
    return 0;
}

static void free_arrays(SimulationArrays *arr)
{
#define FREE_FIELD(name)  \
    do                    \
    {                     \
        free(arr->name);  \
        arr->name = NULL; \
    } while (0)

    FREE_FIELD(LUXI);
    FREE_FIELD(LASI);
    FREE_FIELD(C6i);
    FREE_FIELD(C12i);
    FREE_FIELD(C6e);
    FREE_FIELD(C12e);
    FREE_FIELD(TETR);
    FREE_FIELD(LACI);
    FREE_FIELD(RFP);
    FREE_FIELD(GFP);
    FREE_FIELD(LUXI_next);
    FREE_FIELD(LASI_next);
    FREE_FIELD(C6i_next);
    FREE_FIELD(C12i_next);
    FREE_FIELD(C6e_next);
    FREE_FIELD(C12e_next);
    FREE_FIELD(TETR_next);
    FREE_FIELD(LACI_next);
    FREE_FIELD(RFP_next);
    FREE_FIELD(GFP_next);
    FREE_FIELD(lap_C6);
    FREE_FIELD(lap_C12);
    FREE_FIELD(pLT);
    FREE_FIELD(pLL);
#ifdef WITH_GUI
    free(arr->mag_history);
#endif

#undef FREE_FIELD
}

/* -------------------------------------------------------------------------- */
/* Simulation primitives                                                      */
/* -------------------------------------------------------------------------- */
typedef struct
{
    double dm_c6;
    double dm_c12;
    double de_c6;
    double de_c12;
    double c6_conc;
    double c12_conc;
    double c6_noise_frac;
    double c12_noise_frac;
} SimulationInput;

typedef struct
{
    double final_time_minutes;
    double min_conv_minutes;
    double tolerance;
    size_t max_steps;
    size_t stats_stride;
    size_t field_stride;
    const char *output_dir;
    int replicate_id;
    size_t batch_id;
    size_t job_id;
    bool verbose;
    bool enable_gui;
    size_t viz_stride;
} SimulationRunConfig;

static void laplacian_pbc(const double *field, double *out)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < GRID_N; ++i)
    {
        const int ip = (i + 1 == GRID_N) ? 0 : i + 1;
        const int im = (i == 0) ? (GRID_N - 1) : i - 1;
        for (int j = 0; j < GRID_N; ++j)
        {
            const int jp = (j + 1 == GRID_N) ? 0 : j + 1;
            const int jm = (j == 0) ? (GRID_N - 1) : j - 1;
            const size_t center = idx(i, j);
            const double lap = 0.5 * field[idx(ip, j)] + 0.5 * field[idx(im, j)] +
                               0.5 * field[idx(i, jp)] + 0.5 * field[idx(i, jm)] +
                               0.25 * field[idx(ip, jp)] + 0.25 * field[idx(ip, jm)] +
                               0.25 * field[idx(im, jp)] + 0.25 * field[idx(im, jm)] -
                               3.0 * field[center];
            out[center] = lap * DH2_INV;
        }
    }
}

static void initialize_state(SimulationArrays *arr, const SimulationInput *input, RngState *rng)
{
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double noise_c6 = rng_normal(rng, input->c6_conc, input->c6_conc * input->c6_noise_frac);
        const double noise_c12 = rng_normal(rng, input->c12_conc, input->c12_conc * input->c12_noise_frac);
        arr->C6e[k] = clip_nonnegative(noise_c6);
        arr->C12e[k] = clip_nonnegative(noise_c12);
    }
}

static void write_metadata(const char *dir_path, const SimulationInput *input, const SimulationRunConfig *cfg)
{
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/simulation_parameters.txt", dir_path);
    FILE *f = fopen(file_path, "w");
    if (!f)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", file_path, strerror(errno));
        return;
    }
    fprintf(f, "K_1R = %.6e\n", K_1R);
    fprintf(f, "K_1S = %.6e\n", K_1S);
    fprintf(f, "K_GR = %.6e\n", K_GR);
    fprintf(f, "K_GS = %.6e\n", K_GS);
    fprintf(f, "K_TET = %.6e\n", K_TET);
    fprintf(f, "K_LAC = %.6e\n", K_LAC);
    fprintf(f, "n_R = %.2f\n", n_R);
    fprintf(f, "n_S = %.2f\n", n_S);
    fprintf(f, "Domain size (micrometers): %.1f\n", DOMAIN_SIZE_UM);
    fprintf(f, "Grid points: %d\n", GRID_N);
    fprintf(f, "Grid spacing (micrometers): %.4f\n", DH);
    fprintf(f, "Dm_c6 = %.6e\n", input->dm_c6);
    fprintf(f, "Dm_c12 = %.6e\n", input->dm_c12);
    fprintf(f, "Dext_c6 = %.6e\n", input->de_c6);
    fprintf(f, "Dext_c12 = %.6e\n", input->de_c12);
    fprintf(f, "C6_noise_frac = %.3f\n", input->c6_noise_frac);
    fprintf(f, "C12_noise_frac = %.3f\n", input->c12_noise_frac);
    fprintf(f, "Final_time_minutes = %.2f\n", cfg->final_time_minutes);
    fprintf(f, "Min_convergence_minutes = %.2f\n", cfg->min_conv_minutes);
    fprintf(f, "Stats_stride_steps = %zu\n", cfg->stats_stride);
    fprintf(f, "Field_stride_steps = %zu\n", cfg->field_stride);
    fprintf(f, "Field_file = fields_f32.bin\n");
    fprintf(f, "Field_record = uint64 step + float32 GFP[N*N] + float32 RFP[N*N]\n");
    fclose(f);
}

typedef struct
{
    FILE *stats_csv;
    FILE *fields_bin;
    FILE *fields_index_csv;
    float *gfp_f32;
    float *rfp_f32;
    size_t field_count;
} RunWriters;

static void close_run_writers(RunWriters *writers)
{
    if (!writers)
    {
        return;
    }
    if (writers->stats_csv)
    {
        fclose(writers->stats_csv);
        writers->stats_csv = NULL;
    }
    if (writers->fields_bin)
    {
        fclose(writers->fields_bin);
        writers->fields_bin = NULL;
    }
    if (writers->fields_index_csv)
    {
        fclose(writers->fields_index_csv);
        writers->fields_index_csv = NULL;
    }
    free(writers->gfp_f32);
    free(writers->rfp_f32);
    writers->gfp_f32 = NULL;
    writers->rfp_f32 = NULL;
    writers->field_count = 0;
}

static int init_run_writers(const char *dir_path, RunWriters *writers)
{
    memset(writers, 0, sizeof(*writers));
    char path_stats[1024];
    char path_fields[1024];
    char path_index[1024];
    snprintf(path_stats, sizeof(path_stats), "%s/timeseries_stats.csv", dir_path);
    snprintf(path_fields, sizeof(path_fields), "%s/fields_f32.bin", dir_path);
    snprintf(path_index, sizeof(path_index), "%s/fields_index.csv", dir_path);

    writers->stats_csv = fopen(path_stats, "w");
    if (!writers->stats_csv)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", path_stats, strerror(errno));
        close_run_writers(writers);
        return -1;
    }
    fprintf(writers->stats_csv, "step,minutes,convergence,gfp_mean,rfp_mean,magnetization\n");

    writers->fields_bin = fopen(path_fields, "wb");
    if (!writers->fields_bin)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", path_fields, strerror(errno));
        close_run_writers(writers);
        return -1;
    }

    writers->fields_index_csv = fopen(path_index, "w");
    if (!writers->fields_index_csv)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", path_index, strerror(errno));
        close_run_writers(writers);
        return -1;
    }
    fprintf(writers->fields_index_csv, "snapshot_id,step,minutes,byte_offset\n");

    writers->gfp_f32 = (float *)malloc(sizeof(float) * GRID_TOTAL);
    writers->rfp_f32 = (float *)malloc(sizeof(float) * GRID_TOTAL);
    if (!writers->gfp_f32 || !writers->rfp_f32)
    {
        fprintf(stderr, "Failed to allocate float32 conversion buffers.\n");
        close_run_writers(writers);
        return -1;
    }

    const char magic[8] = {'R', 'D', 'I', 'M', 'F', '3', '2', '\0'};
    const uint32_t version = 1;
    const uint32_t grid_n = (uint32_t)GRID_N;
    const uint64_t total_cells = (uint64_t)GRID_TOTAL;
    const double dt = DT;
    fwrite(magic, sizeof(char), 8, writers->fields_bin);
    fwrite(&version, sizeof(uint32_t), 1, writers->fields_bin);
    fwrite(&grid_n, sizeof(uint32_t), 1, writers->fields_bin);
    fwrite(&dt, sizeof(double), 1, writers->fields_bin);
    fwrite(&total_cells, sizeof(uint64_t), 1, writers->fields_bin);
    return 0;
}

static void append_stats_row(RunWriters *writers, size_t step, double minutes, double convergence,
                             double gfp_mean, double rfp_mean, double magnetization)
{
    if (!writers || !writers->stats_csv)
    {
        return;
    }
    fprintf(writers->stats_csv, "%zu,%.6f,%.9e,%.9e,%.9e,%.9e\n",
            step, minutes, convergence, gfp_mean, rfp_mean, magnetization);
}

static void append_field_snapshot_f32(RunWriters *writers, size_t step, double minutes,
                                      const double *gfp, const double *rfp)
{
    if (!writers || !writers->fields_bin || !writers->gfp_f32 || !writers->rfp_f32)
    {
        return;
    }
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        writers->gfp_f32[k] = (float)gfp[k];
        writers->rfp_f32[k] = (float)rfp[k];
    }

    const uint64_t step_u64 = (uint64_t)step;
    const long long byte_offset = (long long)ftello(writers->fields_bin);
    fwrite(&step_u64, sizeof(uint64_t), 1, writers->fields_bin);
    fwrite(writers->gfp_f32, sizeof(float), GRID_TOTAL, writers->fields_bin);
    fwrite(writers->rfp_f32, sizeof(float), GRID_TOTAL, writers->fields_bin);

    if (writers->fields_index_csv)
    {
        fprintf(writers->fields_index_csv, "%zu,%zu,%.6f,%lld\n",
                writers->field_count, step, minutes, byte_offset);
    }
    writers->field_count += 1;
}

static double advance_step(SimulationArrays *arr, const SimulationInput *input)
{
    laplacian_pbc(arr->C6e, arr->lap_C6);
    laplacian_pbc(arr->C12e, arr->lap_C12);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        arr->pLT[k] = promoter_function(arr->C6i[k], arr->TETR[k], K_GR, K_2R, K_1R, K_TET, n_R, n_REP1);
        arr->pLL[k] = promoter_function(arr->C12i[k], arr->LACI[k], K_GS, K_2S, K_1R, K_LAC, n_S, n_REP2);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double luxi = arr->LUXI[k];
        const double lasi = arr->LASI[k];
        const double c6i = arr->C6i[k];
        const double c12i = arr->C12i[k];
        const double c6e = arr->C6e[k];
        const double c12e = arr->C12e[k];
        const double tetr = arr->TETR[k];
        const double laci = arr->LACI[k];
        const double rfp = arr->RFP[k];
        const double gfp = arr->GFP[k];

        const double enzyme_c6 = (k_LUXI * luxi) / (KM_ENZ + luxi + 1e-12);
        const double enzyme_c12 = (k_LASI * lasi) / (KM_ENZ + lasi + 1e-12);

        const double expr_LT = (arr->pLT[k] * k_tx + b_0) * ktr_over_dmrna;
        const double expr_LL = (arr->pLL[k] * k_tx + b_1) * ktr_over_dmrna;

        const double c6i_next = c6i + (enzyme_c6 + input->dm_c6 * (c6e - c6i) - d_HSLi * c6i) * DT;
        const double c12i_next = c12i + (enzyme_c12 + input->dm_c12 * (c12e - c12i) - d_HSLi * c12i) * DT;
        const double c6e_next = c6e + (input->dm_c6 * (c6i - c6e) + input->de_c6 * arr->lap_C6[k] - d_HSLe * c6e) * DT;
        const double c12e_next = c12e + (input->dm_c12 * (c12i - c12e) + input->de_c12 * arr->lap_C12[k] - d_HSLe * c12e) * DT;

        const double tetr_next = tetr + (expr_LL - d_p * tetr) * DT;
        const double laci_next = laci + (expr_LT - d_p * laci) * DT;
        const double luxi_next = luxi + (expr_LT - d_p * luxi) * DT;
        const double lasi_next = lasi + (expr_LL - d_p * lasi) * DT;
        const double rfp_next = rfp + (expr_LT - d_p * rfp) * DT;
        const double gfp_next = gfp + (expr_LL - d_p * gfp) * DT;

        arr->C6i_next[k] = clip_nonnegative(c6i_next);
        arr->C12i_next[k] = clip_nonnegative(c12i_next);
        arr->C6e_next[k] = clip_nonnegative(c6e_next);
        arr->C12e_next[k] = clip_nonnegative(c12e_next);
        arr->TETR_next[k] = clip_nonnegative(tetr_next);
        arr->LACI_next[k] = clip_nonnegative(laci_next);
        arr->LUXI_next[k] = clip_nonnegative(luxi_next);
        arr->LASI_next[k] = clip_nonnegative(lasi_next);
        arr->RFP_next[k] = clip_nonnegative(rfp_next);
        arr->GFP_next[k] = clip_nonnegative(gfp_next);
    }

    swap_arrays(&arr->C6i, &arr->C6i_next);
    swap_arrays(&arr->C12i, &arr->C12i_next);
    swap_arrays(&arr->C6e, &arr->C6e_next);
    swap_arrays(&arr->C12e, &arr->C12e_next);
    swap_arrays(&arr->TETR, &arr->TETR_next);
    swap_arrays(&arr->LACI, &arr->LACI_next);
    swap_arrays(&arr->LUXI, &arr->LUXI_next);
    swap_arrays(&arr->LASI, &arr->LASI_next);
    swap_arrays(&arr->RFP, &arr->RFP_next);
    swap_arrays(&arr->GFP, &arr->GFP_next);

    const double mean_curr = array_mean(arr->GFP, GRID_TOTAL);
    const double mean_prev = array_mean(arr->GFP_next, GRID_TOTAL);
    return fabs(mean_curr - mean_prev);
}

static int run_simulation(const SimulationInput *input, const SimulationRunConfig *cfg)
{
    SimulationArrays arrays;
    if (allocate_arrays(&arrays) != 0)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return -1;
    }

    const double dt = DT;
    size_t max_steps = cfg->max_steps;
    if (max_steps == 0)
    {
        const double total_minutes = cfg->final_time_minutes;
        max_steps = (size_t)fmax(1.0, total_minutes / dt);
    }

#ifdef WITH_GUI
    ViewerHandle *viewer = NULL;
    uint8_t *viz_rgb = NULL;
    size_t viz_stride = cfg->viz_stride ? cfg->viz_stride : 500;
    size_t history_index = 0;
    if (cfg->enable_gui)
    {
        viewer = viewer_create(GRID_N, GRID_N, viz_stride);
        if (!viewer)
        {
            fprintf(stderr, "Failed to initialize GUI viewer, continuing headless.\n");
        }
        else
        {
            viz_rgb = (uint8_t *)malloc(GRID_TOTAL * 3);
            if (!viz_rgb)
            {
                fprintf(stderr, "Failed to allocate visualization buffer, disabling GUI.\n");
                viewer_destroy(viewer);
                viewer = NULL;
            }
            else
            {
                arrays.mag_history = (double *)calloc(max_steps, sizeof(double));
                if (!arrays.mag_history)
                {
                    fprintf(stderr, "Failed to allocate magnetization history, disabling GUI timeline.\n");
                }
            }
        }
    }
#else
    if (cfg->enable_gui)
    {
        fprintf(stderr, "This binary was compiled without GUI support.\n");
    }
#endif

    RngState rng;
    const uint64_t seed = ((uint64_t)time(NULL) << 32) ^
                          ((uint64_t)cfg->replicate_id << 24) ^
                          ((uint64_t)cfg->batch_id << 16) ^
                          ((uint64_t)cfg->job_id << 8) ^
                          (uint64_t)(uintptr_t)&arrays;
    rng_seed(&rng, seed);
    initialize_state(&arrays, input, &rng);

    char sim_dir[1024];
    const time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", &tm_now);
    snprintf(sim_dir, sizeof(sim_dir), "%s/simulation_%s_rep%d_n%d_dm%.4f_de%.4f_c6e%.6f_job%zu",
             cfg->output_dir, timestamp, cfg->replicate_id, GRID_N, input->dm_c6, input->de_c6, input->c6_conc, cfg->job_id);

    if (ensure_dir(cfg->output_dir) != 0 || ensure_dir(sim_dir) != 0)
    {
        fprintf(stderr, "Could not create directory %s: %s\n", sim_dir, strerror(errno));
        free_arrays(&arrays);
        return -1;
    }
    write_metadata(sim_dir, input, cfg);
    RunWriters writers;
    if (init_run_writers(sim_dir, &writers) != 0)
    {
        free_arrays(&arrays);
        return -1;
    }

    double convergence = 1.0;
    size_t step = 0;
    const double min_conv_minutes = cfg->min_conv_minutes;
    double magnetization = 0.0;
    while ((convergence > cfg->tolerance || (step * dt) < min_conv_minutes) && step < max_steps)
    {
#ifdef WITH_GUI
        if (viewer)
        {
            viewer_poll(viewer);
        }
#endif
        convergence = advance_step(&arrays, input);
        ++step;
        magnetization = compute_magnetization_stats(arrays.GFP, arrays.RFP);
#ifdef WITH_GUI
        if (viewer)
        {
            if (arrays.mag_history && history_index < max_steps)
            {
                arrays.mag_history[history_index++] = magnetization;
            }
        }
#endif
        if (cfg->stats_stride > 0 && (step % cfg->stats_stride == 0))
        {
            const double gfp_mean = array_mean(arrays.GFP, GRID_TOTAL);
            const double rfp_mean = array_mean(arrays.RFP, GRID_TOTAL);
            append_stats_row(&writers, step, step * dt, convergence, gfp_mean, rfp_mean, magnetization);
            if (cfg->verbose)
            {
                printf("[rep %d | batch %zu | job %zu] step %zu, convergence %.3e\n",
                       cfg->replicate_id, cfg->batch_id, cfg->job_id, step, convergence);
            }
        }
        if (cfg->field_stride > 0 && (step % cfg->field_stride == 0))
        {
            append_field_snapshot_f32(&writers, step, step * dt, arrays.GFP, arrays.RFP);
        }

#ifdef WITH_GUI
        if (viewer && viz_rgb)
        {
            size_t requested_stride = viewer_current_stride(viewer);
            if (requested_stride == 0)
            {
                requested_stride = 1;
            }
            viz_stride = requested_stride;
            if (step % viz_stride == 0)
            {
                FieldStats g_stats;
                FieldStats r_stats;
                compose_rgb_frame(arrays.GFP, arrays.RFP, viz_rgb, &g_stats, &r_stats, NULL);
                ViewerFrameMeta meta = {
                    .step = step,
                    .sim_minutes = step * DT,
                    .convergence = convergence,
                    .gfp_mean = g_stats.mean,
                    .rfp_mean = r_stats.mean,
                    .magnetization = magnetization,
                    .mag_history = arrays.mag_history,
                    .mag_history_count = history_index,
                    .mag_history_capacity = max_steps,
                    .minutes_per_step = DT,
                    .total_minutes = cfg->final_time_minutes};
                viewer_submit_frame(viewer, viz_rgb, GRID_TOTAL * 3, meta);
            }
            if (viewer_wants_exit(viewer))
            {
                if (cfg->verbose)
                {
                    printf("Viewer requested termination, stopping simulation early.\n");
                }
                break;
            }
        }
#endif
    }

    if (step > 0 && cfg->stats_stride > 0 && (step % cfg->stats_stride != 0))
    {
        const double gfp_mean = array_mean(arrays.GFP, GRID_TOTAL);
        const double rfp_mean = array_mean(arrays.RFP, GRID_TOTAL);
        append_stats_row(&writers, step, step * dt, convergence, gfp_mean, rfp_mean, magnetization);
    }
    if (step > 0 && cfg->field_stride > 0 && (step % cfg->field_stride != 0))
    {
        append_field_snapshot_f32(&writers, step, step * dt, arrays.GFP, arrays.RFP);
    }

    if (cfg->verbose)
    {
        printf("Simulation finished after %zu steps (conv=%.3e) -> %s\n",
               step, convergence, sim_dir);
    }

#ifdef WITH_GUI
    if (viewer)
    {
        viewer_destroy(viewer);
    }
    free(viz_rgb);
#endif

    close_run_writers(&writers);
    free_arrays(&arrays);
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Default exploration sets                                                   */
/* -------------------------------------------------------------------------- */
// static const double C6_CONCS[] = {
//     0.0114866, 0.0023850, 0.0012480, 0.0005676, 0.0003431,
//     0.0002694, 0.0002331, 0.0001623, 0.0001487, 0.0001388};
static const double C6_CONCS[] =
    {0.011479302483635982,
     0.002377898894988739,
     0.0012412329321509008,
     0.0005615533502252251,
     0.00033788358671171174,
     0.0002648754222972973,
     0.00022903645833333334,
     0.00015970228040540542,
     0.00014640695382882885,
     0.00013677153716216216};

static const double DM_COEFFS[] = {
    0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 3.0, 5.0, 10.0};
static const double DEXT_COEFFS[] = {
    0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0};
static const int GRID_SIZE_LIST[] = {32, 64, 128, 256};

static size_t num_c6_concs(void)
{
    return sizeof(C6_CONCS) / sizeof(double);
}

static size_t num_dm_coeffs(void)
{
    return sizeof(DM_COEFFS) / sizeof(double);
}

static size_t num_dext_coeffs(void)
{
    return sizeof(DEXT_COEFFS) / sizeof(double);
}

static size_t num_grid_sizes(void)
{
    return sizeof(GRID_SIZE_LIST) / sizeof(int);
}

/* -------------------------------------------------------------------------- */
/* CLI handling                                                               */
/* -------------------------------------------------------------------------- */
typedef struct
{
    int replicas;
    double tolerance;
    double min_conv_minutes;
    int grid_points;
    size_t stats_stride;
    size_t field_stride;
    size_t max_steps;
    double final_time_minutes;
    bool verbose;
    char output_dir[512];
    bool gui;
    size_t viz_stride;
    size_t dm_index;
    size_t de_index;
    double c6_noise_frac;
    double c12_noise_frac;
} CliOptions;

static void set_default_cli(CliOptions *opts)
{
    opts->replicas = 3;
    opts->tolerance = 1e-8;
    opts->stats_stride = SNAPSHOT_STRIDE_DEFAULT;
    opts->field_stride = 15 * SNAPSHOT_STRIDE_DEFAULT;
    opts->max_steps = 0;
    opts->final_time_minutes = 24.0 * 60.0;
    opts->min_conv_minutes = 720.0;
    opts->grid_points = 256;
    opts->verbose = true;
    strncpy(opts->output_dir, OUTPUT_DIR_DEFAULT, sizeof(opts->output_dir));
    opts->gui = false;
    opts->viz_stride = 10;
    opts->dm_index = 0;
    opts->de_index = 0;
    opts->c6_noise_frac = 0.25;
    opts->c12_noise_frac = 0.25;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --replicas N          Number of replica batches (default 3)\n");
    printf("  --tol VALUE           Convergence tolerance (default 1e-8)\n");
    printf("  --min-conv-time MIN   Minimum simulated minutes before convergence check can stop (default 720)\n");
    printf("  --grid-size N         Ignored in gradient mode (uses 32,64,128,256)\n");
    printf("  --stats-stride N      Scalar stats stride in steps (default 1000)\n");
    printf("  --field-stride N      Full-field stride in steps (default 15000)\n");
    printf("  --snap STRIDE         Alias for --stats-stride\n");
    printf("  --max-steps N         Safety cap on steps (default derived from final time)\n");
    printf("  --final-time MIN      Nominal end time in minutes (default 1440)\n");
    printf("  --output DIR          Directory for simulation folders (default %s)\n", OUTPUT_DIR_DEFAULT);
    printf("  --gui                 Ignored in gradient mode (headless sweep only)\n");
    printf("  --viz-stride N        Ignored in gradient mode\n");
    printf("  --dm-index N          Select membrane diffusion coefficient index for the sweep\n");
    printf("  --de-index N          Ignored in gradient mode (all DEXT_COEFFS are swept)\n");
    printf("  --c6-noise-frac F     Std dev as fraction of mean for initial C6e noise (default 0.25)\n");
    printf("  --c12-noise-frac F    Std dev as fraction of mean for initial C12e noise (default 0.25)\n");
    printf("  --quiet               Reduce stdout logging\n");
    printf("  --help                Show this message\n");
    printf("Enable OpenMP by compiling with -fopenmp and exporting OMP_NUM_THREADS.\n");
}

static bool parse_cli(int argc, char **argv, CliOptions *opts)
{
    set_default_cli(opts);
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--replicas") == 0 && i + 1 < argc)
        {
            opts->replicas = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc)
        {
            opts->tolerance = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--min-conv-time") == 0 && i + 1 < argc)
        {
            opts->min_conv_minutes = atof(argv[++i]);
            if (opts->min_conv_minutes < 0.0)
            {
                opts->min_conv_minutes = 0.0;
            }
        }
        else if (strcmp(argv[i], "--grid-size") == 0 && i + 1 < argc)
        {
            opts->grid_points = atoi(argv[++i]);
            if (opts->grid_points < 4)
            {
                opts->grid_points = 4;
            }
        }
        else if (strcmp(argv[i], "--stats-stride") == 0 && i + 1 < argc)
        {
            opts->stats_stride = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--field-stride") == 0 && i + 1 < argc)
        {
            opts->field_stride = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--snap") == 0 && i + 1 < argc)
        {
            opts->stats_stride = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc)
        {
            opts->max_steps = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--final-time") == 0 && i + 1 < argc)
        {
            opts->final_time_minutes = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            strncpy(opts->output_dir, argv[++i], sizeof(opts->output_dir));
            opts->output_dir[sizeof(opts->output_dir) - 1] = '\0';
        }
        else if (strcmp(argv[i], "--gui") == 0)
        {
            opts->gui = true;
        }
        else if (strcmp(argv[i], "--viz-stride") == 0 && i + 1 < argc)
        {
            opts->viz_stride = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--dm-index") == 0 && i + 1 < argc)
        {
            opts->dm_index = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--de-index") == 0 && i + 1 < argc)
        {
            opts->de_index = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--c6-noise-frac") == 0 && i + 1 < argc)
        {
            opts->c6_noise_frac = atof(argv[++i]);
            if (opts->c6_noise_frac < 0.0)
            {
                opts->c6_noise_frac = 0.0;
            }
        }
        else if (strcmp(argv[i], "--c12-noise-frac") == 0 && i + 1 < argc)
        {
            opts->c12_noise_frac = atof(argv[++i]);
            if (opts->c12_noise_frac < 0.0)
            {
                opts->c12_noise_frac = 0.0;
            }
        }
        else if (strcmp(argv[i], "--quiet") == 0)
        {
            opts->verbose = false;
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return false;
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (opts->replicas < 1)
    {
        opts->replicas = 1;
    }
    if (opts->viz_stride == 0)
    {
        opts->viz_stride = 500;
    }
    if (opts->gui)
    {
        if (opts->verbose)
        {
            fprintf(stderr, "--gui is ignored in gradient mode; running headless.\n");
        }
        opts->gui = false;
    }
    return true;
}

/* -------------------------------------------------------------------------- */
/* Main driver                                                                */
/* -------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    CliOptions cli;
    if (!parse_cli(argc, argv, &cli))
    {
        return 1;
    }

    if (ensure_dir(cli.output_dir) != 0)
    {
        fprintf(stderr, "Could not create output directory %s: %s\n", cli.output_dir, strerror(errno));
        return 1;
    }

    const size_t dm_count = num_dm_coeffs();
    const size_t de_count = num_dext_coeffs();
    if (dm_count != num_c6_concs())
    {
        fprintf(stderr, "Mismatch between DM coefficients and concentration sets.\n");
        return 1;
    }

    const size_t dm_idx = (cli.dm_index < dm_count) ? cli.dm_index : dm_count - 1;

    double *c6_grid = NULL;
    const size_t c6_points = build_c6_grid(dm_idx, &c6_grid);
    if (c6_points == 0 || !c6_grid)
    {
        fprintf(stderr, "Failed to build C6 gradient grid.\n");
        return 1;
    }

    const double dm_val = DM_COEFFS[dm_idx];
    const size_t total_param_tasks = de_count * c6_points;
    if (total_param_tasks == 0)
    {
        fprintf(stderr, "No Dext/C6 tasks to run.\n");
        free(c6_grid);
        return 1;
    }

    for (int replica = 0; replica < cli.replicas; ++replica)
    {
        const size_t n_sizes = num_grid_sizes();
        for (size_t n_idx = 0; n_idx < n_sizes; ++n_idx)
        {
            set_grid_geometry(GRID_SIZE_LIST[n_idx]);
            char out_dir[512];
            snprintf(out_dir, sizeof(out_dir), "%s/n%d", cli.output_dir, GRID_N);
            ensure_dir(cli.output_dir);
            ensure_dir(out_dir);

            if (cli.verbose)
            {
                printf("Replica %d: lattice n=%d, sweeping %zu Dext x %zu C6 points\n",
                       replica, GRID_N, de_count, c6_points);
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (size_t task_idx = 0; task_idx < total_param_tasks; ++task_idx)
            {
                const size_t de_idx = task_idx / c6_points;
                const size_t c_idx = task_idx % c6_points;
                const double de_val = DEXT_COEFFS[de_idx];
                SimulationInput input = {
                    .dm_c6 = dm_val,
                    .dm_c12 = dm_val,
                    .de_c6 = de_val,
                    .de_c12 = de_val,
                    .c6_conc = c6_grid[c_idx],
                    .c12_conc = 0.0,
                    .c6_noise_frac = cli.c6_noise_frac,
                    .c12_noise_frac = cli.c12_noise_frac};

                SimulationRunConfig cfg = {
                    .final_time_minutes = cli.final_time_minutes,
                    .min_conv_minutes = cli.min_conv_minutes,
                    .tolerance = cli.tolerance,
                    .max_steps = cli.max_steps,
                    .stats_stride = cli.stats_stride,
                    .field_stride = cli.field_stride,
                    .output_dir = out_dir,
                    .replicate_id = replica,
                    .batch_id = de_idx,
                    .job_id = task_idx,
                    .verbose = cli.verbose,
                    .enable_gui = false,
                    .viz_stride = cli.viz_stride};

                const int status = run_simulation(&input, &cfg);
                if (status != 0)
                {
                    fprintf(stderr, "Simulation failed for Dext idx %zu, C6 idx %zu at grid %d.\n",
                            de_idx, c_idx, GRID_N);
                }
            }
        }
    }

    free(c6_grid);
    return 0;
}
