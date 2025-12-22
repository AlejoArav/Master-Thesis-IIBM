#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ViewerHandle ViewerHandle;

typedef struct
{
    size_t step;
    double sim_minutes;
    double convergence;
    double gfp_mean;
    double magnetization;
    double rfp_mean;
    double occupancy_mean;
    double birth_rate;
    double death_rate;
    double init_occupancy;
    const double *mag_history;
    size_t mag_history_count;
    size_t mag_history_capacity;
    double minutes_per_step;
    double total_minutes;
} ViewerFrameMeta;

#ifdef __cplusplus
extern "C"
{
#endif

    ViewerHandle *viewer_create(size_t width, size_t height, size_t initial_stride);
    void viewer_destroy(ViewerHandle *viewer);
    void viewer_poll(ViewerHandle *viewer);
    void viewer_submit_frame(ViewerHandle *viewer,
                             const uint8_t *expr_rgb, size_t expr_byte_count,
                             const uint8_t *occ_rgb, size_t occ_byte_count,
                             ViewerFrameMeta meta);
    bool viewer_wants_exit(const ViewerHandle *viewer);
    size_t viewer_current_stride(const ViewerHandle *viewer);

#ifdef __cplusplus
}
#endif
