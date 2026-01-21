#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <sys/stat.h>
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

#define N 256
#define DOMAIN_SIZE 256.0
#define DT 0.001
#define SNAPSHOT_STRIDE_DEFAULT 1000
#define OUTPUT_DIR_DEFAULT "cp_rdim_outputs"

static const double DH = DOMAIN_SIZE / (double)N;
static const double DH2_INV = 1.0 / (DH * DH);
static const size_t GRID_TOTAL = (size_t)N * (size_t)N;

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

static FieldStats compute_field_stats_masked(const double *field, const double *cells, size_t *count_out)
{
    double min_v = DBL_MAX;
    double max_v = -DBL_MAX;
    double sum = 0.0;
    size_t count = 0;
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        if (cells && cells[k] > 0.5)
        {
            const double v = field[k];
            ++count;
            sum += v;
            if (v < min_v)
            {
                min_v = v;
            }
            if (v > max_v)
            {
                max_v = v;
            }
        }
    }
    if (count_out)
    {
        *count_out = count;
    }
    if (count == 0)
    {
        return compute_field_stats(field);
    }
    FieldStats stats = {min_v, max_v, sum / (double)count};
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

static void compose_rgb_frame(const double *gfp, const double *rfp, const double *cells, uint8_t *rgb,
                              FieldStats *out_gfp, FieldStats *out_rfp,
                              double *out_mag, double *out_occ)
{
    size_t occ_count = 0;
    FieldStats g_stats = compute_field_stats_masked(gfp, cells, &occ_count);
    FieldStats r_stats = compute_field_stats_masked(rfp, cells, NULL);
    double occ_sum = 0.0;
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double occ = cells ? cells[k] : 0.0;
        occ_sum += occ;
        if (occ < 0.5)
        {
            rgb[3 * k + 0] = 0;
            rgb[3 * k + 1] = 0;
            rgb[3 * k + 2] = 0;
            continue;
        }
        const uint8_t r_enc = encode_uint8(rfp[k], r_stats.min, r_stats.max);
        const uint8_t g_enc = encode_uint8(gfp[k], g_stats.min, g_stats.max);
        /* toggle-like coloring: only dominant channel is shown */
        if (r_enc >= g_enc)
        {
            rgb[3 * k + 0] = r_enc;
            rgb[3 * k + 1] = 0;
        }
        else
        {
            rgb[3 * k + 0] = 0;
            rgb[3 * k + 1] = g_enc;
        }
        /* blue channel no longer encodes occupancy */
        rgb[3 * k + 2] = 0;
    }
    if (out_mag)
    {
        *out_mag = compute_magnetization(gfp, rfp);
    }
    if (out_occ)
    {
        *out_occ = occ_sum / (double)GRID_TOTAL;
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

static const double C6_CONCS[] = {
    0.01148090, 0.00237850, 0.00124500, 0.00056350, 0.00033850,
    0.00026450, 0.00022800, 0.00015900, 0.00014450, 0.00013450};
static const double DM_COEFFS[] = {
    0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 3.0, 5.0, 10.0};
static const double DEXT_COEFFS[] = {
    0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0};

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

/* RNG utilities (xorshift128+) */
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

/* Helpers */
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

static inline double sample_trunc_normal(RngState *rng, double mean, double sd, double min_v, double max_v)
{
    double val;
    do
    {
        val = rng_normal(rng, mean, sd);
    } while (val < min_v || val > max_v);
    return val;
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

static inline size_t idx(int i, int j)
{
    const int wrapped_i = (i % N + N) % N;
    const int wrapped_j = (j % N + N) % N;
    return (size_t)wrapped_i * (size_t)N + (size_t)wrapped_j;
}

static void swap_arrays(double **a, double **b)
{
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Data containers */
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
    double *cells;
    double *cells_next;
    double *volume;
    double *volume_next;
    double *lap_C6;
    double *lap_C12;
    double *pLT;
    double *pLL;
    double *nutrient;
    double *nutrient_next;
    double *lap_nutrient;
    double *division_target;
#ifdef WITH_GUI
    double *mag_history;
    double *gfp_history;
    double *rfp_history;
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
    ALLOC_FIELD(cells);
    ALLOC_FIELD(cells_next);
    ALLOC_FIELD(volume);
    ALLOC_FIELD(volume_next);
    ALLOC_FIELD(lap_C6);
    ALLOC_FIELD(lap_C12);
    ALLOC_FIELD(pLT);
    ALLOC_FIELD(pLL);
    ALLOC_FIELD(nutrient);
    ALLOC_FIELD(nutrient_next);
    ALLOC_FIELD(lap_nutrient);
    ALLOC_FIELD(division_target);
#ifdef WITH_GUI
    arr->mag_history = NULL;
    arr->gfp_history = NULL;
    arr->rfp_history = NULL;
#endif

#undef ALLOC_FIELD
    return 0;
}

/* helpers for division/movement */
static inline void copy_intracellular(const SimulationArrays *arr, size_t src, size_t dst)
{
    arr->LUXI[dst] = arr->LUXI[src];
    arr->LASI[dst] = arr->LASI[src];
    arr->C6i[dst] = arr->C6i[src];
    arr->C12i[dst] = arr->C12i[src];
    arr->TETR[dst] = arr->TETR[src];
    arr->LACI[dst] = arr->LACI[src];
    arr->RFP[dst] = arr->RFP[src];
    arr->GFP[dst] = arr->GFP[src];
}

static inline void clear_intracellular(SimulationArrays *arr, size_t idx)
{
    arr->LUXI[idx] = 0.0;
    arr->LASI[idx] = 0.0;
    arr->C6i[idx] = 0.0;
    arr->C12i[idx] = 0.0;
    arr->TETR[idx] = 0.0;
    arr->LACI[idx] = 0.0;
    arr->RFP[idx] = 0.0;
    arr->GFP[idx] = 0.0;
}

static inline void move_cell_state(SimulationArrays *arr, size_t from, size_t to)
{
    arr->cells[to] = arr->cells[from];
    arr->cells[from] = 0.0;

    arr->volume[to] = arr->volume[from];
    arr->volume[from] = 0.0;
    arr->division_target[to] = arr->division_target[from];
    arr->division_target[from] = 0.0;

    copy_intracellular(arr, from, to);
    clear_intracellular(arr, from);
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
    FREE_FIELD(cells);
    FREE_FIELD(cells_next);
    FREE_FIELD(volume);
    FREE_FIELD(volume_next);
    FREE_FIELD(lap_C6);
    FREE_FIELD(lap_C12);
    FREE_FIELD(pLT);
    FREE_FIELD(pLL);
    FREE_FIELD(nutrient);
    FREE_FIELD(nutrient_next);
    FREE_FIELD(lap_nutrient);
    FREE_FIELD(division_target);
#ifdef WITH_GUI
    free(arr->mag_history);
    free(arr->gfp_history);
    free(arr->rfp_history);
#endif

#undef FREE_FIELD
}

typedef struct
{
    double dm_c6;
    double dm_c12;
    double de_c6;
    double de_c12;
    double c6_conc;
    double c12_conc;
    double nutrient_diffusion;
    double nutrient_decay;
    double nutrient_init;
    double nutrient_uptake;
} SimulationInput;

typedef struct
{
    double birth_rate;
    double init_occupancy;
    bool seed_center;
    double growth_rate;     /* exponential growth rate per minute */
    double division_volume; /* critical volume at which division is attempted */
    int division_mode;      /* 0=B (Moore search with retries), 1=D (random Moore pick), 2=E (random radius pick) */
    size_t division_radius; /* used by mode E; radius in Chebyshev distance layers */
    double move_prob;       /* probability to attempt a move when division fails (modes D/E) */
    double kappa;           /* nutrient half-saturation */
    double xi;              /* toxin/public-good sensitivity */
    double division_mean;
    double division_sd;
    double division_min;
    double division_max;
    size_t seed_count;      /* explicit number of initial occupied sites (random) */
    double min_convergence_time; /* minutes before convergence stopping is considered */
} ContactParams;

typedef struct
{
    double final_time_minutes;
    double tolerance;
    size_t max_steps;
    size_t snapshot_stride;
    const char *output_dir;
    int replicate_id;
    size_t batch_id;
    size_t job_id;
    bool verbose;
    bool enable_gui;
    size_t viz_stride;
    ContactParams contact;
} SimulationRunConfig;

/* Core operations */
static void laplacian_pbc(const double *field, double *out)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < N; ++i)
    {
        const int ip = (i + 1 == N) ? 0 : i + 1;
        const int im = (i == 0) ? (N - 1) : i - 1;
        for (int j = 0; j < N; ++j)
        {
            const int jp = (j + 1 == N) ? 0 : j + 1;
            const int jm = (j == 0) ? (N - 1) : j - 1;
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

static void initialize_state(SimulationArrays *arr, const SimulationInput *input,
                             const ContactParams *contact, RngState *rng)
{
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        const double noise_c6 = rng_normal(rng, input->c6_conc, input->c6_conc * 0.25);
        const double noise_c12 = rng_normal(rng, input->c12_conc, input->c12_conc * 0.25);
        arr->C6e[k] = clip_nonnegative(noise_c6);
        arr->C12e[k] = clip_nonnegative(noise_c12);
        const double occ = rng_uniform(rng) < contact->init_occupancy ? 1.0 : 0.0;
        arr->cells[k] = occ;
        arr->cells_next[k] = occ;
        const double v0 = occ > 0.5 ? 1.0 : 0.0;
        arr->volume[k] = v0;
        arr->volume_next[k] = v0;
        arr->nutrient[k] = input->nutrient_init;
        arr->nutrient_next[k] = input->nutrient_init;
        const double div_target = sample_trunc_normal(rng, contact->division_mean, contact->division_sd,
                                                      contact->division_min, contact->division_max);
        arr->division_target[k] = div_target;
    }

    if (contact->seed_center)
    {
        memset(arr->cells, 0, GRID_TOTAL * sizeof(double));
        memset(arr->cells_next, 0, GRID_TOTAL * sizeof(double));
        memset(arr->volume, 0, GRID_TOTAL * sizeof(double));
        memset(arr->volume_next, 0, GRID_TOTAL * sizeof(double));
        memset(arr->division_target, 0, GRID_TOTAL * sizeof(double));
        const int center = N / 2;
        const size_t center_idx = idx(center, center);
        arr->cells[center_idx] = 1.0;
        arr->cells_next[center_idx] = 1.0;
        arr->volume[center_idx] = 1.0;
        arr->volume_next[center_idx] = 1.0;
        const double div_target = sample_trunc_normal(rng, contact->division_mean, contact->division_sd,
                                                      contact->division_min, contact->division_max);
        arr->division_target[center_idx] = div_target;
    }
    else if (contact->seed_count > 0)
    {
        memset(arr->cells, 0, GRID_TOTAL * sizeof(double));
        memset(arr->cells_next, 0, GRID_TOTAL * sizeof(double));
        memset(arr->volume, 0, GRID_TOTAL * sizeof(double));
        memset(arr->volume_next, 0, GRID_TOTAL * sizeof(double));
        memset(arr->division_target, 0, GRID_TOTAL * sizeof(double));
        const size_t max_seed = contact->seed_count > GRID_TOTAL ? GRID_TOTAL : contact->seed_count;
        const size_t center = N / 2;
        /* collect offsets around center sorted by manhattan then chebyshev distance */
        size_t placed = 0;
        const size_t radius = (size_t)ceil(sqrt((double)max_seed));
        typedef struct
        {
            int dx;
            int dy;
            int manhattan;
            int cheb;
        } Offset;
        const size_t max_offsets = (2 * radius + 1) * (2 * radius + 1);
        Offset *offsets = (Offset *)malloc(max_offsets * sizeof(Offset));
        size_t off_count = 0;
        for (int dx = -(int)radius; dx <= (int)radius; ++dx)
        {
            for (int dy = -(int)radius; dy <= (int)radius; ++dy)
            {
                if (dx == 0 && dy == 0)
                {
                    continue;
                }
                offsets[off_count++] = (Offset){dx, dy, abs(dx) + abs(dy), (int)fmax(abs(dx), abs(dy))};
            }
        }
        /* sort offsets by manhattan, then chebyshev, then |dx|,|dy| to prioritize near-center, cardinal first */
        for (size_t a = 0; a + 1 < off_count; ++a)
        {
            for (size_t b = a + 1; b < off_count; ++b)
            {
                Offset oa = offsets[a];
                Offset ob = offsets[b];
                bool swap = false;
                if (oa.manhattan > ob.manhattan)
                    swap = true;
                else if (oa.manhattan == ob.manhattan)
                {
                    if (oa.cheb > ob.cheb)
                        swap = true;
                    else if (oa.cheb == ob.cheb)
                    {
                        if (abs(oa.dx) + abs(oa.dy) > abs(ob.dx) + abs(ob.dy))
                            swap = true;
                    }
                }
                if (swap)
                {
                    offsets[a] = ob;
                    offsets[b] = oa;
                }
            }
        }
        /* place center first */
        const size_t center_idx = idx((int)center, (int)center);
        arr->cells[center_idx] = 1.0;
        arr->cells_next[center_idx] = 1.0;
        arr->volume[center_idx] = 1.0;
        arr->volume_next[center_idx] = 1.0;
        arr->division_target[center_idx] = sample_trunc_normal(rng, contact->division_mean, contact->division_sd,
                                                               contact->division_min, contact->division_max);
        placed = 1;
        for (size_t m = 0; m < off_count && placed < max_seed; ++m)
        {
            const int i = (int)center + offsets[m].dx;
            const int j = (int)center + offsets[m].dy;
            const size_t k = idx(i, j);
            if (arr->cells[k] > 0.5)
            {
                continue;
            }
            arr->cells[k] = 1.0;
            arr->cells_next[k] = 1.0;
            arr->volume[k] = 1.0;
            arr->volume_next[k] = 1.0;
            arr->division_target[k] = sample_trunc_normal(rng, contact->division_mean, contact->division_sd,
                                                          contact->division_min, contact->division_max);
            ++placed;
        }
        free(offsets);
    }
}

static void write_metadata(const char *dir_path, const SimulationInput *input,
                           const ContactParams *contact)
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
    fprintf(f, "Dm_c6 = %.6e\n", input->dm_c6);
    fprintf(f, "Dm_c12 = %.6e\n", input->dm_c12);
    fprintf(f, "Dext_c6 = %.6e\n", input->de_c6);
    fprintf(f, "Dext_c12 = %.6e\n", input->de_c12);
    fprintf(f, "Initial_C6e = %.6e\n", input->c6_conc);
    fprintf(f, "Initial_C12e = %.6e\n", input->c12_conc);
    fprintf(f, "Nutrient_diffusion = %.6e\n", input->nutrient_diffusion);
    fprintf(f, "Nutrient_decay = %.6e\n", input->nutrient_decay);
    fprintf(f, "Nutrient_init = %.6e\n", input->nutrient_init);
    fprintf(f, "Nutrient_uptake = %.6e\n", input->nutrient_uptake);
    fprintf(f, "Birth_rate = %.6e\n", contact->birth_rate);
    fprintf(f, "Init_occupancy = %.3f\n", contact->init_occupancy);
    fprintf(f, "Seed_center = %s\n", contact->seed_center ? "true" : "false");
    fprintf(f, "Seed_count = %zu\n", contact->seed_count);
    fprintf(f, "Growth_rate = %.6e\n", contact->growth_rate);
    fprintf(f, "Division_volume = %.3f\n", contact->division_volume);
    fprintf(f, "Division_mode = %d\n", contact->division_mode);
    fprintf(f, "Division_radius = %zu\n", contact->division_radius);
    fprintf(f, "Move_prob = %.3f\n", contact->move_prob);
    fprintf(f, "Kappa = %.6e\n", contact->kappa);
    fprintf(f, "Xi = %.6e\n", contact->xi);
    fprintf(f, "Division_mean = %.3f\n", contact->division_mean);
    fprintf(f, "Division_sd = %.3f\n", contact->division_sd);
    fprintf(f, "Division_min = %.3f\n", contact->division_min);
    fprintf(f, "Division_max = %.3f\n", contact->division_max);
    fprintf(f, "Min_convergence_time = %.3f\n", contact->min_convergence_time);
    fclose(f);
}

static void write_snapshot(const char *dir_path, size_t snapshot_idx,
                           const double *gfp, const double *rfp)
{
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/gfp_rfp_%zu.bin", dir_path, snapshot_idx);
    FILE *f = fopen(file_path, "wb");
    if (!f)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", file_path, strerror(errno));
        return;
    }
    const uint64_t total = (uint64_t)GRID_TOTAL;
    fwrite(&total, sizeof(uint64_t), 1, f);
    fwrite(gfp, sizeof(double), GRID_TOTAL, f);
    fwrite(rfp, sizeof(double), GRID_TOTAL, f);
    fclose(f);
}

static void write_occupancy_snapshot(const char *dir_path, size_t snapshot_idx,
                                     const double *cells)
{
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/cells_%zu.bin", dir_path, snapshot_idx);
    FILE *f = fopen(file_path, "wb");
    if (!f)
    {
        fprintf(stderr, "Could not open %s for writing: %s\n", file_path, strerror(errno));
        return;
    }
    const uint64_t total = (uint64_t)GRID_TOTAL;
    fwrite(&total, sizeof(uint64_t), 1, f);
    uint8_t *buffer = malloc(GRID_TOTAL);
    if (!buffer)
    {
        fprintf(stderr, "Failed to allocate occupancy buffer\n");
        fclose(f);
        return;
    }
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        buffer[k] = cells[k] > 0.5 ? 1 : 0;
    }
    fwrite(buffer, sizeof(uint8_t), GRID_TOTAL, f);
    free(buffer);
    fclose(f);
}

static double advance_step(SimulationArrays *arr, const SimulationInput *input, const ContactParams *params)
{
    /* global occupancy fraction (0-1) to mimic dilution in a finite chamber */
    const double colony_fraction = array_mean(arr->cells, GRID_TOTAL);

    laplacian_pbc(arr->C6e, arr->lap_C6);
    laplacian_pbc(arr->C12e, arr->lap_C12);
    laplacian_pbc(arr->nutrient, arr->lap_nutrient);

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
        const double occ = arr->cells[k] > 0.5 ? 1.0 : 0.0;
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
        const double nutrient = arr->nutrient[k];

        const double enzyme_c6 = occ * (k_LUXI * luxi) / (KM_ENZ + luxi + 1e-12);
        const double enzyme_c12 = occ * (k_LASI * lasi) / (KM_ENZ + lasi + 1e-12);

        const double expr_LT = occ * ((arr->pLT[k] * k_tx + b_0) * ktr_over_dmrna);
        const double expr_LL = occ * ((arr->pLL[k] * k_tx + b_1) * ktr_over_dmrna);

        const double c6i_next = c6i + (enzyme_c6 + occ * input->dm_c6 * (c6e - c6i) - d_HSLi * c6i) * DT;
        const double c12i_next = c12i + (enzyme_c12 + occ * input->dm_c12 * (c12e - c12i) - d_HSLi * c12i) * DT;
        /* membrane exchange scaled by global occupancy to represent dilution */
        const double c6e_next = c6e + (colony_fraction * occ * input->dm_c6 * (c6i - c6e) + input->de_c6 * arr->lap_C6[k] - d_HSLe * c6e) * DT;
        const double c12e_next = c12e + (colony_fraction * occ * input->dm_c12 * (c12i - c12e) + input->de_c12 * arr->lap_C12[k] - d_HSLe * c12e) * DT;

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

        const double uptake = occ ? input->nutrient_uptake * arr->volume[k] * (nutrient / (params->kappa + nutrient + 1e-12)) : 0.0;
        const double nutrient_next = nutrient + (input->nutrient_diffusion * arr->lap_nutrient[k] -
                                                 input->nutrient_decay * nutrient - uptake) * DT;
        arr->nutrient_next[k] = nutrient_next < 0.0 ? 0.0 : nutrient_next;
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
    swap_arrays(&arr->nutrient, &arr->nutrient_next);

    const double mean_curr = array_mean(arr->GFP, GRID_TOTAL);
    const double mean_prev = array_mean(arr->GFP_next, GRID_TOTAL);
    return fabs(mean_curr - mean_prev);
}

static void contact_process_step(SimulationArrays *arr, const ContactParams *params, RngState *rng)
{
    const double dt = DT;

    /* Step 1: exponential volume growth for occupied cells */
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const size_t k = idx(i, j);
            const bool occupied = arr->cells[k] > 0.5;
            double occ = occupied ? 1.0 : 0.0;

            if (occupied)
            {
                const double v = arr->volume[k];
                const double n = arr->nutrient[k];
                const double n_factor = n / (params->kappa + n + 1e-12);
                const double toxin_factor = fmax(0.0, 1.0 - params->xi * 0.0); /* xi>0 if toxin coupled later */
                const double dl = params->growth_rate * v * n_factor * toxin_factor * dt;
                if (dl > 0.0)
                {
                    arr->volume[k] = v + dl;
                }
            }

            arr->cells_next[k] = occ;
        }
    }

    swap_arrays(&arr->cells, &arr->cells_next);

    /* Step 2: division attempts (mode-dependent) and optional movement */
    static const int moore_di[8] = {1, 1, 1, 0, 0, -1, -1, -1};
    static const int moore_dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const size_t k = idx(i, j);
            if (arr->cells[k] < 0.5)
            {
                continue;
            }

            double v = arr->volume[k];
            const double target_len = arr->division_target[k] > 0.0 ? arr->division_target[k] : params->division_volume;
            if (v < target_len)
            {
                continue;
            }

            size_t daughter_idx = (size_t)-1;
            if (params->division_mode == 0)
            {
                /* Mode B (default): random ordering of Moore neighbors with retries until an empty one is found */
                int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                for (int m = 7; m > 0; --m)
                {
                    const int r = (int)(rng_uniform(rng) * (double)(m + 1));
                    const int tmp = order[m];
                    order[m] = order[r];
                    order[r] = tmp;
                }
                for (int m = 0; m < 8; ++m)
                {
                    const int ni = i + moore_di[order[m]];
                    const int nj = j + moore_dj[order[m]];
                    const size_t nk = idx(ni, nj);
                    if (arr->cells[nk] < 0.5)
                    {
                        daughter_idx = nk;
                        break;
                    }
                }
            }
            else if (params->division_mode == 1)
            {
                /* Mode D: single random Moore neighbor, no retries */
                const int r = (int)(rng_uniform(rng) * 8.0);
                const int ni = i + moore_di[r];
                const int nj = j + moore_dj[r];
                const size_t nk = idx(ni, nj);
                if (arr->cells[nk] < 0.5)
                {
                    daughter_idx = nk;
                }
            }
            else /* params->division_mode == 2 */
            {
                /* Mode E: random offset within Chebyshev radius (layers), no retries */
                size_t radius = params->division_radius < 1 ? 1 : params->division_radius;
                int dx = 0;
                int dy = 0;
                /* sample until we get a non-zero offset within radius */
                do
                {
                    dx = (int)(rng_uniform(rng) * (double)(2 * radius + 1)) - (int)radius;
                    dy = (int)(rng_uniform(rng) * (double)(2 * radius + 1)) - (int)radius;
                } while ((dx == 0 && dy == 0) || (abs(dx) > (int)radius) || (abs(dy) > (int)radius));

                const int ni = i + dx;
                const int nj = j + dy;
                const size_t nk = idx(ni, nj);
                if (arr->cells[nk] < 0.5)
                {
                    daughter_idx = nk;
                }
            }

            if (daughter_idx != (size_t)-1)
            {
                /* symmetric division: reset mother and daughter volumes to half of division volume */
                const double new_target_m = sample_trunc_normal(rng, params->division_mean, params->division_sd,
                                                                params->division_min, params->division_max);
                const double new_target_d = sample_trunc_normal(rng, params->division_mean, params->division_sd,
                                                                params->division_min, params->division_max);
                arr->division_target[k] = new_target_m;
                arr->division_target[daughter_idx] = new_target_d;
                const double v_old = arr->volume[k];
                const double v_m = 0.5 * new_target_m;
                const double v_d = 0.5 * new_target_d;
                arr->volume[k] = v_m;
                arr->volume[daughter_idx] = v_d;

                /* occupy daughter site */
                arr->cells[daughter_idx] = 1.0;

                /* deterministic partition: conserve mass -> concentration = total_mass / total_new_volume */
                const double total_vol = v_m + v_d;
                const double inv_total_vol = (total_vol > 0.0) ? (1.0 / total_vol) : 0.0;
                const double luxi_mass = arr->LUXI[k] * v_old;
                const double lasi_mass = arr->LASI[k] * v_old;
                const double c6i_mass = arr->C6i[k] * v_old;
                const double c12i_mass = arr->C12i[k] * v_old;
                const double tetr_mass = arr->TETR[k] * v_old;
                const double laci_mass = arr->LACI[k] * v_old;
                const double rfp_mass = arr->RFP[k] * v_old;
                const double gfp_mass = arr->GFP[k] * v_old;

                const double conc_luxi = luxi_mass * inv_total_vol;
                const double conc_lasi = lasi_mass * inv_total_vol;
                const double conc_c6i = c6i_mass * inv_total_vol;
                const double conc_c12i = c12i_mass * inv_total_vol;
                const double conc_tetr = tetr_mass * inv_total_vol;
                const double conc_laci = laci_mass * inv_total_vol;
                const double conc_rfp = rfp_mass * inv_total_vol;
                const double conc_gfp = gfp_mass * inv_total_vol;

                arr->LUXI[k] = conc_luxi;
                arr->LASI[k] = conc_lasi;
                arr->C6i[k] = conc_c6i;
                arr->C12i[k] = conc_c12i;
                arr->TETR[k] = conc_tetr;
                arr->LACI[k] = conc_laci;
                arr->RFP[k] = conc_rfp;
                arr->GFP[k] = conc_gfp;

                arr->LUXI[daughter_idx] = conc_luxi;
                arr->LASI[daughter_idx] = conc_lasi;
                arr->C6i[daughter_idx] = conc_c6i;
                arr->C12i[daughter_idx] = conc_c12i;
                arr->TETR[daughter_idx] = conc_tetr;
                arr->LACI[daughter_idx] = conc_laci;
                arr->RFP[daughter_idx] = conc_rfp;
                arr->GFP[daughter_idx] = conc_gfp;
                /* extracellular fields are left untouched; diffusion/exchange handles them */
                /* optional movement after successful division for modes D/E */
                if (params->division_mode != 0 && params->move_prob > 0.0 && rng_uniform(rng) <= params->move_prob)
                {
                    size_t target_idx = (size_t)-1;
                    if (params->division_mode == 1)
                    {
                        const int r = (int)(rng_uniform(rng) * 8.0);
                        const int ni = i + moore_di[r];
                        const int nj = j + moore_dj[r];
                        const size_t nk = idx(ni, nj);
                        if (arr->cells[nk] < 0.5)
                        {
                            target_idx = nk;
                        }
                    }
                    else /* mode E movement */
                    {
                        size_t radius = params->division_radius < 1 ? 1 : params->division_radius;
                        int dx = 0;
                        int dy = 0;
                        do
                        {
                            dx = (int)(rng_uniform(rng) * (double)(2 * radius + 1)) - (int)radius;
                            dy = (int)(rng_uniform(rng) * (double)(2 * radius + 1)) - (int)radius;
                        } while ((dx == 0 && dy == 0) || (abs(dx) > (int)radius) || (abs(dy) > (int)radius));

                        const int ni = i + dx;
                        const int nj = j + dy;
                        const size_t nk = idx(ni, nj);
                        if (arr->cells[nk] < 0.5)
                        {
                            target_idx = nk;
                        }
                    }

                    if (target_idx != (size_t)-1)
                    {
                        move_cell_state(arr, k, target_idx);
                    }
                }
            }
        }
    }
}

static void apply_cell_mask(SimulationArrays *arr)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < GRID_TOTAL; ++k)
    {
        if (arr->cells[k] < 0.5)
        {
            arr->LUXI[k] = 0.0;
            arr->LASI[k] = 0.0;
            arr->C6i[k] = 0.0;
            arr->C12i[k] = 0.0;
            arr->TETR[k] = 0.0;
            arr->LACI[k] = 0.0;
            arr->RFP[k] = 0.0;
            arr->GFP[k] = 0.0;
        }
    }
}

static int run_simulation(const SimulationInput *input, const SimulationRunConfig *cfg)
{
    SimulationArrays arrays;
    if (allocate_arrays(&arrays) != 0)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return -1;
    }

    size_t max_steps = cfg->max_steps;
    if (max_steps == 0)
    {
        max_steps = (size_t)fmax(1.0, cfg->final_time_minutes / DT);
    }

#ifdef WITH_GUI
    ViewerHandle *viewer = NULL;
    uint8_t *viz_rgb = NULL;
    size_t viz_stride = cfg->viz_stride ? cfg->viz_stride : 500;
    size_t history_index = 0;
    if (cfg->enable_gui)
    {
        viewer = viewer_create(N, N, viz_stride);
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
                arrays.gfp_history = (double *)calloc(max_steps, sizeof(double));
                arrays.rfp_history = (double *)calloc(max_steps, sizeof(double));
                if (!arrays.mag_history)
                {
                    fprintf(stderr, "Failed to allocate magnetization history, disabling timeline.\n");
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
    initialize_state(&arrays, input, &cfg->contact, &rng);

    char sim_dir[1024];
    const time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", &tm_now);
    snprintf(sim_dir, sizeof(sim_dir), "%s/simulation_%s_rep%d_dm%.4f_de%.4f_job%zu",
             cfg->output_dir, timestamp, cfg->replicate_id, input->dm_c6, input->de_c6, cfg->job_id);

    if (ensure_dir(cfg->output_dir) != 0 || ensure_dir(sim_dir) != 0)
    {
        fprintf(stderr, "Could not create directory %s: %s\n", sim_dir, strerror(errno));
        free_arrays(&arrays);
        return -1;
    }
    write_metadata(sim_dir, input, &cfg->contact);

    double convergence = 1.0;
    size_t step = 0;
    size_t snapshot_counter = 0;
    while (convergence > cfg->tolerance && step < max_steps)
    {
#ifdef WITH_GUI
        if (viewer)
        {
            viewer_poll(viewer);
        }
#endif
        convergence = advance_step(&arrays, input, &cfg->contact);
        contact_process_step(&arrays, &cfg->contact, &rng);
        apply_cell_mask(&arrays);
        const double sim_minutes = step * DT;
        if (sim_minutes < cfg->contact.min_convergence_time)
        {
            convergence = cfg->tolerance + 1.0; /* skip convergence check before threshold */
        }
        ++step;

#ifdef WITH_GUI
        double magnetization = 0.0;
        if (viewer)
        {
            magnetization = compute_magnetization(arrays.GFP, arrays.RFP);
            if (history_index < max_steps)
            {
                const double g_mean_hist = array_mean(arrays.GFP, GRID_TOTAL);
                const double r_mean_hist = array_mean(arrays.RFP, GRID_TOTAL);
                if (arrays.mag_history)
                {
                    arrays.mag_history[history_index] = magnetization;
                }
                if (arrays.gfp_history)
                {
                    arrays.gfp_history[history_index] = g_mean_hist;
                }
                if (arrays.rfp_history)
                {
                    arrays.rfp_history[history_index] = r_mean_hist;
                }
                ++history_index;
            }
        }
#endif

        if (cfg->snapshot_stride > 0 && (step % cfg->snapshot_stride == 0))
        {
            write_snapshot(sim_dir, snapshot_counter, arrays.GFP, arrays.RFP);
            write_occupancy_snapshot(sim_dir, snapshot_counter, arrays.cells);
            ++snapshot_counter;
            if (cfg->verbose)
            {
                printf("[rep %d | batch %zu | job %zu] step %zu, conv %.3e\n",
                       cfg->replicate_id, cfg->batch_id, cfg->job_id, step, convergence);
            }
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
                double frame_mag = magnetization;
                double occ_mean = 0.0;
                compose_rgb_frame(arrays.GFP, arrays.RFP, arrays.cells, viz_rgb,
                                  &g_stats, &r_stats, &frame_mag, &occ_mean);

                static uint8_t *occ_rgb = NULL;
                static size_t occ_capacity = 0;
                const size_t occ_bytes = GRID_TOTAL * 3;
                if (occ_capacity < occ_bytes)
                {
                    free(occ_rgb);
                    occ_rgb = (uint8_t *)malloc(occ_bytes);
                    if (!occ_rgb)
                    {
                        fprintf(stderr, "Failed to allocate occupancy RGB buffer for viewer.\n");
                        break;
                    }
                    occ_capacity = occ_bytes;
                }
                for (size_t k = 0; k < GRID_TOTAL; ++k)
                {
                    const uint8_t v = (arrays.cells[k] > 0.5) ? 255 : 0;
                    occ_rgb[3 * k + 0] = v;
                    occ_rgb[3 * k + 1] = v;
                    occ_rgb[3 * k + 2] = v;
                }
                    ViewerFrameMeta meta = {
                        .step = step,
                        .sim_minutes = step * DT,
                        .convergence = convergence,
                        .gfp_mean = g_stats.mean,
                        .magnetization = frame_mag,
                        .rfp_mean = r_stats.mean,
                        .occupancy_mean = occ_mean,
                        .birth_rate = cfg->contact.birth_rate,
                        .init_occupancy = cfg->contact.init_occupancy,
                        .mag_history = arrays.mag_history,
                        .mag_history_count = history_index,
                        .mag_history_capacity = max_steps,
                        .gfp_history = arrays.gfp_history,
                        .rfp_history = arrays.rfp_history,
                        .gfp_history_count = history_index,
                        .rfp_history_count = history_index,
                        .expr_history_capacity = max_steps,
                        .minutes_per_step = DT,
                        .total_minutes = cfg->final_time_minutes};
                viewer_submit_frame(viewer,
                                    viz_rgb, GRID_TOTAL * 3,
                                    occ_rgb, occ_bytes,
                                    meta);
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

    free_arrays(&arrays);
    return 0;
}

/* CLI */
typedef struct
{
    int replicas;
    double tolerance;
    size_t snapshot_stride;
    size_t max_steps;
    double final_time_minutes;
    bool verbose;
    char output_dir[512];
    double birth_rate;
    double init_occupancy;
    double growth_rate;
    double division_volume;
    size_t dm_index;
    size_t de_index;
    bool gui;
    size_t viz_stride;
    bool seed_center;
    bool no_convergence_stop;
    double c6_conc_override;
    double c12_conc_override;
    int division_mode;
    size_t division_radius;
    double move_prob;
    double nutrient_init;
    double nutrient_diffusion;
    double nutrient_decay;
    double nutrient_uptake;
    double kappa;
    double division_mean;
    double division_sd;
    double division_min;
    double division_max;
    size_t seed_count;
    double min_convergence_time;
} CliOptions;

static void set_default_cli(CliOptions *opts)
{
    opts->replicas = 1;
    opts->tolerance = 1e-6;
    opts->snapshot_stride = SNAPSHOT_STRIDE_DEFAULT;
    opts->max_steps = 0;
    opts->final_time_minutes = 12.0 * 60.0;
    opts->verbose = true;
    strncpy(opts->output_dir, OUTPUT_DIR_DEFAULT, sizeof(opts->output_dir));
    opts->birth_rate = 0.05;
    opts->init_occupancy = 0.5;
    opts->growth_rate = log(2.0) / 30.0; /* ~0.0231 1/min: 30 min division from V=1 to V=2 */
    opts->division_volume = 2.0;
    opts->dm_index = (size_t)-1;
    opts->de_index = (size_t)-1;
    opts->gui = false;
    opts->viz_stride = 500;
    opts->seed_center = false;
    opts->no_convergence_stop = false;
    opts->c6_conc_override = NAN;
    opts->c12_conc_override = NAN;
    opts->division_mode = 0;     /* default: Mode B (Moore with retries) */
    opts->division_radius = 1;   /* used only for Mode E */
    opts->move_prob = 1.0;       /* used for Modes D/E */
    opts->nutrient_init = 1.0;
    opts->nutrient_diffusion = 500.0 / 60.0; /* 500 um^2/hr -> per minute */
    opts->nutrient_decay = 0.0;
    opts->nutrient_uptake = 2.0 / 60.0;      /* 2 hr^-1 -> per minute */
    opts->kappa = 0.333;
    opts->division_mean = 4.0;
    opts->division_sd = 0.3;
    opts->division_min = 2.5;
    opts->division_max = 5.5;
    opts->seed_count = 0;
    opts->min_convergence_time = 0.0;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --replicas N          Number of replica batches (default 1)\n");
    printf("  --tol VALUE           Convergence tolerance (default 1e-6; ignored if --no-convergence-stop is set)\n");
    printf("  --snap STRIDE         Snapshot stride in steps (default 1000)\n");
    printf("  --max-steps N         Safety cap on steps (default derived from final time)\n");
    printf("  --final-time MIN      Nominal end time in minutes (default 720)\n");
    printf("  --output DIR          Directory for simulation folders (default %s)\n", OUTPUT_DIR_DEFAULT);
    printf("  --birth-rate VALUE    Reserved legacy contact-process parameter (currently unused) (default 0.05)\n");
    printf("  --init-occupancy F    Initial occupied fraction (0-1, default 0.5)\n");
    printf("  --growth-rate VALUE   Exponential volume growth rate per minute (default 0.03)\n");
    printf("  --division-volume V   Volume threshold for division (default 2.0)\n");
    printf("  --seed-center         Override occupancy to a single central cell\n");
    printf("  --seed-count N        Override occupancy to exactly N random occupied sites\n");
    printf("  --no-convergence-stop Run until max-steps/final-time without checking convergence\n");
    printf("  --gui                 Enable SDL2 + ImGui live viewer (requires GUI build)\n");
    printf("  --viz-stride N        Steps between viewer refreshes (default 500)\n");
    printf("  --dm-index N          Restrict to membrane diffusion index N (optional)\n");
    printf("  --de-index N          Restrict to external diffusion index N (optional)\n");
    printf("  --c6-conc VALUE       Override initial extracellular C6 concentration (default table-driven)\n");
    printf("  --c12-conc VALUE      Override initial extracellular C12 concentration (default 0)\n");
    printf("  --division-mode M     Division neighborhood mode: B (default, Moore with retries), D (random Moore pick), E (random radius pick)\n");
    printf("  --division-radius N   Radius (layers) for Mode E (Chebyshev distance, default 1)\n");
    printf("  --move-prob P         Probability of attempting movement when division fails in Modes D/E (default 1.0)\n");
    printf("  --min-conv-time MIN   Do not check convergence until this simulated minute (default 0)\n");
    printf("  --nutrient-init V     Initial nutrient concentration (default 1.0)\n");
    printf("  --nutrient-diffusion D   Nutrient diffusion coefficient (default 0.01)\n");
    printf("  --nutrient-decay R    Nutrient decay rate (default 0)\n");
    printf("  --nutrient-uptake R   Nutrient uptake rate per unit length (default 0.1)\n");
    printf("  --kappa V             Nutrient half-saturation constant (default 0.5)\n");
    printf("  --division-mean V     Mean division length (um) (default 4.0)\n");
    printf("  --division-sd V       Std dev division length (um) (default 0.3)\n");
    printf("  --division-min V      Min (truncation) division length (default 2.5)\n");
    printf("  --division-max V      Max (truncation) division length (default 5.5)\n");
    printf("  --quiet               Reduce stdout logging\n");
    printf("  --help                Show this message\n");
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
        else if (strcmp(argv[i], "--snap") == 0 && i + 1 < argc)
        {
            opts->snapshot_stride = (size_t)strtoull(argv[++i], NULL, 10);
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
        else if (strcmp(argv[i], "--birth-rate") == 0 && i + 1 < argc)
        {
            opts->birth_rate = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--init-occupancy") == 0 && i + 1 < argc)
        {
            opts->init_occupancy = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--growth-rate") == 0 && i + 1 < argc)
        {
            opts->growth_rate = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--division-volume") == 0 && i + 1 < argc)
        {
            opts->division_volume = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--dm-index") == 0 && i + 1 < argc)
        {
            opts->dm_index = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--de-index") == 0 && i + 1 < argc)
        {
            opts->de_index = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--gui") == 0)
        {
            opts->gui = true;
        }
        else if (strcmp(argv[i], "--viz-stride") == 0 && i + 1 < argc)
        {
            opts->viz_stride = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--seed-center") == 0)
        {
            opts->seed_center = true;
        }
        else if (strcmp(argv[i], "--min-conv-time") == 0 && i + 1 < argc)
        {
            opts->min_convergence_time = atof(argv[++i]);
            if (opts->min_convergence_time < 0.0)
            {
                opts->min_convergence_time = 0.0;
            }
        }
        else if (strcmp(argv[i], "--seed-count") == 0 && i + 1 < argc)
        {
            opts->seed_count = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--division-mode") == 0 && i + 1 < argc)
        {
            const char m = argv[++i][0];
            if (m == 'B' || m == 'b')
            {
                opts->division_mode = 0;
            }
            else if (m == 'D' || m == 'd')
            {
                opts->division_mode = 1;
            }
            else if (m == 'E' || m == 'e')
            {
                opts->division_mode = 2;
            }
            else
            {
                fprintf(stderr, "Unknown division mode '%c', using default B.\n", m);
                opts->division_mode = 0;
            }
        }
        else if (strcmp(argv[i], "--division-radius") == 0 && i + 1 < argc)
        {
            opts->division_radius = (size_t)strtoull(argv[++i], NULL, 10);
            if (opts->division_radius < 1)
            {
                opts->division_radius = 1;
            }
        }
        else if (strcmp(argv[i], "--move-prob") == 0 && i + 1 < argc)
        {
            opts->move_prob = atof(argv[++i]);
            if (opts->move_prob < 0.0)
            {
                opts->move_prob = 0.0;
            }
        }
        else if (strcmp(argv[i], "--nutrient-init") == 0 && i + 1 < argc)
        {
            opts->nutrient_init = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--nutrient-diffusion") == 0 && i + 1 < argc)
        {
            opts->nutrient_diffusion = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--nutrient-decay") == 0 && i + 1 < argc)
        {
            opts->nutrient_decay = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--nutrient-uptake") == 0 && i + 1 < argc)
        {
            opts->nutrient_uptake = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--kappa") == 0 && i + 1 < argc)
        {
            opts->kappa = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--division-mean") == 0 && i + 1 < argc)
        {
            opts->division_mean = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--division-sd") == 0 && i + 1 < argc)
        {
            opts->division_sd = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--division-min") == 0 && i + 1 < argc)
        {
            opts->division_min = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--division-max") == 0 && i + 1 < argc)
        {
            opts->division_max = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--c6-conc") == 0 && i + 1 < argc)
        {
            opts->c6_conc_override = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--c12-conc") == 0 && i + 1 < argc)
        {
            opts->c12_conc_override = atof(argv[++i]);
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
    if (opts->snapshot_stride == 0)
    {
        opts->snapshot_stride = SNAPSHOT_STRIDE_DEFAULT;
    }
    if (opts->init_occupancy < 0.0)
    {
        opts->init_occupancy = 0.0;
    }
    if (opts->init_occupancy > 1.0)
    {
        opts->init_occupancy = 1.0;
    }
#ifndef WITH_GUI
    if (opts->gui)
    {
        fprintf(stderr, "Binary was built without GUI support. Recompile with GUI=1 in the Makefile.\n");
        return false;
    }
#endif
    return true;
}

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

    const size_t job_count = num_dm_coeffs();
    const size_t dext_count = num_dext_coeffs();
    const size_t conc_count = num_c6_concs();

    if (job_count != conc_count)
    {
        fprintf(stderr, "Mismatch between DM coefficients and concentration sets.\n");
        return 1;
    }

    if (cli.gui)
    {
        if (cli.replicas != 1)
        {
            fprintf(stderr, "GUI mode supports a single replica; forcing replicas=1.\n");
            cli.replicas = 1;
        }
        const size_t dm_selected = (cli.dm_index != (size_t)-1 && cli.dm_index < job_count) ? cli.dm_index : 0;
        const size_t de_selected = (cli.de_index != (size_t)-1 && cli.de_index < dext_count) ? cli.de_index : 0;
        if (cli.verbose)
        {
            printf("GUI mode: DM[%zu]=%.4f, De[%zu]=%.4f\n",
                   dm_selected, DM_COEFFS[dm_selected], de_selected, DEXT_COEFFS[de_selected]);
        }
        const double c6_init = isnan(cli.c6_conc_override) ? C6_CONCS[dm_selected] : cli.c6_conc_override;
        const double c12_init = isnan(cli.c12_conc_override) ? 0.0 : cli.c12_conc_override;
        SimulationInput input = {
            .dm_c6 = DM_COEFFS[dm_selected],
            .dm_c12 = DM_COEFFS[dm_selected],
            .de_c6 = DEXT_COEFFS[de_selected],
            .de_c12 = DEXT_COEFFS[de_selected],
            .c6_conc = c6_init,
            .c12_conc = c12_init,
            .nutrient_diffusion = cli.nutrient_diffusion,
            .nutrient_decay = cli.nutrient_decay,
            .nutrient_init = cli.nutrient_init,
            .nutrient_uptake = cli.nutrient_uptake};
        SimulationRunConfig cfg = {
            .final_time_minutes = cli.final_time_minutes,
            .tolerance = cli.tolerance,
            .max_steps = cli.max_steps,
            .snapshot_stride = cli.snapshot_stride,
            .output_dir = cli.output_dir,
            .replicate_id = 0,
            .batch_id = de_selected,
            .job_id = dm_selected,
            .verbose = cli.verbose,
            .enable_gui = true,
            .viz_stride = cli.viz_stride,
            .contact = {
                .birth_rate = cli.birth_rate,
                .init_occupancy = cli.init_occupancy,
                .seed_center = cli.seed_center,
                .growth_rate = cli.growth_rate,
                .division_volume = cli.division_volume,
                .division_mode = cli.division_mode,
                .division_radius = cli.division_radius,
                .move_prob = cli.move_prob,
                .kappa = cli.kappa,
                .xi = 0.0,
                .division_mean = cli.division_mean,
                .division_sd = cli.division_sd,
                .division_min = cli.division_min,
                .division_max = cli.division_max,
                .seed_count = cli.seed_count,
                .min_convergence_time = cli.min_convergence_time}};
        const int status = run_simulation(&input, &cfg);
        return status == 0 ? 0 : 1;
    }

    size_t dm_start = 0;
    size_t dm_end = job_count;
    if (cli.dm_index != (size_t)-1)
    {
        dm_start = (cli.dm_index < job_count) ? cli.dm_index : job_count - 1;
        dm_end = dm_start + 1;
    }
    size_t de_start = 0;
    size_t de_end = dext_count;
    if (cli.de_index != (size_t)-1)
    {
        de_start = (cli.de_index < dext_count) ? cli.de_index : dext_count - 1;
        de_end = de_start + 1;
    }

    for (int replica = 0; replica < cli.replicas; ++replica)
    {
        for (size_t de_idx = de_start; de_idx < de_end; ++de_idx)
        {
            const double de_c6 = DEXT_COEFFS[de_idx];
            const double de_c12 = DEXT_COEFFS[de_idx];
            if (cli.verbose)
            {
                printf("Replica %d: Dext pair (%.3f, %.3f)\n", replica, de_c6, de_c12);
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (size_t job = dm_start; job < dm_end; ++job)
            {
                const double c6_init = isnan(cli.c6_conc_override) ? C6_CONCS[job] : cli.c6_conc_override;
                const double c12_init = isnan(cli.c12_conc_override) ? 0.0 : cli.c12_conc_override;
                SimulationInput input = {
                    .dm_c6 = DM_COEFFS[job],
                    .dm_c12 = DM_COEFFS[job],
                    .de_c6 = de_c6,
                    .de_c12 = de_c12,
                    .c6_conc = c6_init,
                    .c12_conc = c12_init,
                    .nutrient_diffusion = cli.nutrient_diffusion,
                    .nutrient_decay = cli.nutrient_decay,
                    .nutrient_init = cli.nutrient_init,
                    .nutrient_uptake = cli.nutrient_uptake};

                SimulationRunConfig cfg = {
                    .final_time_minutes = cli.final_time_minutes,
                    .tolerance = cli.tolerance,
                    .max_steps = cli.max_steps,
                    .snapshot_stride = cli.snapshot_stride,
                    .output_dir = cli.output_dir,
                    .replicate_id = replica,
                    .batch_id = de_idx,
                    .job_id = job,
                    .verbose = cli.verbose,
                    .enable_gui = false,
                    .viz_stride = cli.viz_stride,
                    .contact = {
                        .birth_rate = cli.birth_rate,
                        .init_occupancy = cli.init_occupancy,
                        .seed_center = cli.seed_center,
                        .growth_rate = cli.growth_rate,
                        .division_volume = cli.division_volume,
                        .division_mode = cli.division_mode,
                        .division_radius = cli.division_radius,
                        .move_prob = cli.move_prob,
                        .kappa = cli.kappa,
                        .xi = 0.0,
                        .division_mean = cli.division_mean,
                        .division_sd = cli.division_sd,
                        .division_min = cli.division_min,
                        .division_max = cli.division_max,
                        .seed_count = cli.seed_count,
                        .min_convergence_time = cli.min_convergence_time}};

                const int status = run_simulation(&input, &cfg);
                if (status != 0)
                {
                    fprintf(stderr, "Simulation failed for job %zu in batch %zu.\n", job, de_idx);
                }
            }
            if (cli.verbose)
            {
                printf("Replica %d: finished Dext pair (%.3f, %.3f)\n", replica, de_c6, de_c12);
            }
        }
    }

    return 0;
}
