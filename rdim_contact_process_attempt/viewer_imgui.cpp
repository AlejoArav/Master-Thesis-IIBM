#ifdef WITH_GUI

#include "viewer.h"

#include <SDL.h>

#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_sdlrenderer2.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct ViewerHandle
{
    size_t width;
    size_t height;
    size_t stride;
    bool exit_requested;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *expr_texture;
    SDL_Texture *occ_texture;
    std::vector<uint8_t> expr_frame_buffer;
    std::vector<uint8_t> occ_frame_buffer;
    float display_scale;
    size_t window_size;
    size_t window_offset;
    ViewerFrameMeta last_meta{};
    size_t expr_window_size;
    size_t expr_window_offset;
};

namespace
{

    std::once_flag g_sdl_init_once;
    std::string g_sdl_error;
    std::atomic<bool> g_sdl_ready{false};

    void sdl_init_once()
    {
        SDL_SetHint(SDL_HINT_MAC_BACKGROUND_APP, "1");
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) == 0)
        {
            g_sdl_ready.store(true);
        }
        else
        {
            g_sdl_error = SDL_GetError();
        }
    }

    bool ensure_sdl_ready()
    {
        std::call_once(g_sdl_init_once, sdl_init_once);
        if (!g_sdl_ready.load())
        {
            fprintf(stderr, "SDL initialization failed: %s\n",
                    g_sdl_error.empty() ? "unknown" : g_sdl_error.c_str());
            return false;
        }
        return true;
    }

    struct MagnetPlotContext
    {
        const double *history;
        size_t start;
        size_t total;
    };

    static float magnet_plot_getter(void *data, int idx)
    {
        const auto *ctx = static_cast<const MagnetPlotContext *>(data);
        if (!ctx || !ctx->history || idx < 0)
        {
            return 0.0f;
        }
        const size_t pos = ctx->start + static_cast<size_t>(idx);
        if (pos >= ctx->total)
        {
            return 0.0f;
        }
        return static_cast<float>(ctx->history[pos]);
    }

    static void draw_axis_overlays(const ImVec2 &plot_min, const ImVec2 &plot_max)
    {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        const ImU32 axis_color = IM_COL32(200, 200, 200, 200);
        const float zero_y = plot_min.y + (plot_max.y - plot_min.y) * 0.5f;
        draw_list->AddLine(ImVec2(plot_min.x, zero_y), ImVec2(plot_max.x, zero_y), axis_color, 1.0f);
        const float label_x = plot_min.x - 28.0f;
        const float text_h = ImGui::GetTextLineHeight();
        draw_list->AddText(ImVec2(label_x, plot_min.y - text_h * 0.5f), axis_color, "+1");
        draw_list->AddText(ImVec2(label_x, zero_y - text_h * 0.5f), axis_color, "0");
        draw_list->AddText(ImVec2(label_x, plot_max.y - text_h * 1.2f), axis_color, "-1");
    }

    static void render_magnetization_panel(ViewerHandle *viewer)
    {
        const ViewerFrameMeta &meta = viewer->last_meta;
        if (!meta.mag_history || meta.mag_history_count < 2)
        {
            ImGui::Text("Collecting magnetization history...");
            return;
        }

#if defined(IMGUI_VERSION_NUM) && IMGUI_VERSION_NUM >= 18967
        ImGui::SeparatorText("Magnetization");
#else
        ImGui::Separator();
        ImGui::TextUnformatted("Magnetization");
#endif
        const size_t available = meta.mag_history_count;
        viewer->window_size = std::max<size_t>(1, std::min(viewer->window_size, available));
        size_t max_offset = (available > viewer->window_size) ? (available - viewer->window_size) : 0;
        viewer->window_offset = std::min(viewer->window_offset, max_offset);

        int slider_offset = static_cast<int>(viewer->window_offset);
        const int slider_max = static_cast<int>(max_offset);
        if (ImGui::SliderInt("Timeline", &slider_offset, 0, slider_max))
        {
            viewer->window_offset = static_cast<size_t>(slider_offset);
        }

        int window_size_int = static_cast<int>(viewer->window_size);
        const int max_window = std::max(1, static_cast<int>(available));
        if (ImGui::SliderInt("Window Samples", &window_size_int, 1, max_window))
        {
            viewer->window_size = static_cast<size_t>(window_size_int);
        }
        viewer->window_size = std::max<size_t>(1, std::min(viewer->window_size, available));
        max_offset = (available > viewer->window_size) ? (available - viewer->window_size) : 0;
        viewer->window_offset = std::min(viewer->window_offset, max_offset);

        const size_t display_count = std::min(viewer->window_size, available - viewer->window_offset);
        if (display_count == 0)
        {
            ImGui::Text("Waiting for additional samples...");
            return;
        }

        MagnetPlotContext ctx{meta.mag_history, viewer->window_offset, meta.mag_history_count};
        ImGui::PlotLines("Magnetization", magnet_plot_getter, &ctx,
                         static_cast<int>(display_count), 0, nullptr, -1.0f, 1.0f, ImVec2(0, 120));
        draw_axis_overlays(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());

        const double start_min = meta.minutes_per_step * static_cast<double>(viewer->window_offset);
        const double end_index = viewer->window_offset + display_count - 1;
        const double end_min = meta.minutes_per_step * static_cast<double>(end_index);
        ImGui::Text("Window %.2f-%.2f / %.2f min", start_min, end_min, meta.total_minutes);
    }

    struct ExprPlotContext
    {
        const double *history;
        size_t start;
        size_t total;
    };

    static float expr_plot_getter(void *data, int idx)
    {
        const auto *ctx = static_cast<const ExprPlotContext *>(data);
        if (!ctx || !ctx->history || idx < 0)
        {
            return 0.0f;
        }
        const size_t pos = ctx->start + static_cast<size_t>(idx);
        if (pos >= ctx->total)
        {
            return 0.0f;
        }
        return static_cast<float>(ctx->history[pos]);
    }

    static void render_expression_panel(ViewerHandle *viewer)
    {
        const ViewerFrameMeta &meta = viewer->last_meta;
        if (!meta.gfp_history || !meta.rfp_history || meta.gfp_history_count < 2 || meta.rfp_history_count < 2)
        {
            ImGui::Text("Collecting GFP/RFP history...");
            return;
        }

#if defined(IMGUI_VERSION_NUM) && IMGUI_VERSION_NUM >= 18967
        ImGui::SeparatorText("GFP/RFP means");
#else
        ImGui::Separator();
        ImGui::TextUnformatted("GFP/RFP means");
#endif

        const size_t available = std::min(meta.gfp_history_count, meta.rfp_history_count);
        viewer->expr_window_size = std::max<size_t>(1, std::min(viewer->expr_window_size, available));
        size_t max_offset = (available > viewer->expr_window_size) ? (available - viewer->expr_window_size) : 0;
        viewer->expr_window_offset = std::min(viewer->expr_window_offset, max_offset);

        int slider_offset = static_cast<int>(viewer->expr_window_offset);
        const int slider_max = static_cast<int>(max_offset);
        if (ImGui::SliderInt("Expr Timeline", &slider_offset, 0, slider_max))
        {
            viewer->expr_window_offset = static_cast<size_t>(slider_offset);
        }

        int window_size_int = static_cast<int>(viewer->expr_window_size);
        const int max_window = std::max(1, static_cast<int>(available));
        if (ImGui::SliderInt("Expr Window Samples", &window_size_int, 1, max_window))
        {
            viewer->expr_window_size = static_cast<size_t>(window_size_int);
        }
        viewer->expr_window_size = std::max<size_t>(1, std::min(viewer->expr_window_size, available));
        max_offset = (available > viewer->expr_window_size) ? (available - viewer->expr_window_size) : 0;
        viewer->expr_window_offset = std::min(viewer->expr_window_offset, max_offset);

        const size_t display_count = std::min(viewer->expr_window_size, available - viewer->expr_window_offset);
        if (display_count == 0)
        {
            ImGui::Text("Waiting for additional samples...");
            return;
        }

        ExprPlotContext gctx{meta.gfp_history, viewer->expr_window_offset, meta.gfp_history_count};
        ExprPlotContext rctx{meta.rfp_history, viewer->expr_window_offset, meta.rfp_history_count};

        float min_v = FLT_MAX;
        float max_v = -FLT_MAX;
        for (size_t i = 0; i < display_count; ++i)
        {
            const size_t idx = viewer->expr_window_offset + i;
            if (idx < meta.gfp_history_count)
            {
                const float v = static_cast<float>(meta.gfp_history[idx]);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
            if (idx < meta.rfp_history_count)
            {
                const float v = static_cast<float>(meta.rfp_history[idx]);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
        }
        if (min_v == FLT_MAX || max_v == -FLT_MAX)
        {
            min_v = 0.0f;
            max_v = 1.0f;
        }
        if (fabs(max_v - min_v) < 1e-6f)
        {
            max_v = min_v + 1e-3f;
        }

        ImGui::PlotLines("GFP mean", expr_plot_getter, &gctx,
                         static_cast<int>(display_count), 0, nullptr, min_v, max_v, ImVec2(0, 80));
        ImGui::PlotLines("RFP mean", expr_plot_getter, &rctx,
                         static_cast<int>(display_count), 0, nullptr, min_v, max_v, ImVec2(0, 80));

        const double start_min = meta.minutes_per_step * static_cast<double>(viewer->expr_window_offset);
        const double end_index = viewer->expr_window_offset + display_count - 1;
        const double end_min = meta.minutes_per_step * static_cast<double>(end_index);
        ImGui::Text("Expr window %.2f-%.2f / %.2f min", start_min, end_min, meta.total_minutes);
    }

    void render_ui(ViewerHandle *viewer)
    {
        ImGui_ImplSDL2_NewFrame();
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui::NewFrame();

        const ViewerFrameMeta meta = viewer->last_meta;
        ImGui::Begin("Contact-Process RDIM Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Step: %zu", meta.step);
        ImGui::Text("Time (min): %.2f", meta.sim_minutes);
        ImGui::Text("Convergence: %.3e", meta.convergence);
        ImGui::Text("Mean GFP: %.3e", meta.gfp_mean);
        ImGui::Text("Mean RFP: %.3e", meta.rfp_mean);
        ImGui::Text("Magnetization: %.3f", meta.magnetization);
        ImGui::Text("Mean Occupancy: %.3f", meta.occupancy_mean);
        ImGui::Text("Birth rate (legacy, unused): %.3e", meta.birth_rate);
        ImGui::Text("Initial occupancy: %.2f", meta.init_occupancy);
        int stride = static_cast<int>(viewer->stride);
        if (ImGui::SliderInt("Stride", &stride, 10, 5000))
        {
            viewer->stride = stride > 0 ? static_cast<size_t>(stride) : 1;
        }
        float zoom = viewer->display_scale;
        if (ImGui::SliderFloat("Zoom", &zoom, 1.0f, 4.0f, "%.1fx"))
        {
            viewer->display_scale = zoom;
        }
        render_magnetization_panel(viewer);
        render_expression_panel(viewer);
        ImVec2 image_size(static_cast<float>(viewer->width) * viewer->display_scale,
                          static_cast<float>(viewer->height) * viewer->display_scale);
        ImGui::TextUnformatted("Expression (GFP/RFP)");
        ImGui::Image(viewer->expr_texture, image_size);
        ImGui::Separator();
        ImGui::TextUnformatted("Colonization / occupancy lattice");
        ImGui::Image(viewer->occ_texture, image_size);
        ImGui::End();

        ImGui::Render();
        SDL_SetRenderDrawColor(viewer->renderer, 8, 8, 8, 255);
        SDL_RenderClear(viewer->renderer);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), viewer->renderer);
        SDL_RenderPresent(viewer->renderer);
    }

} // namespace

ViewerHandle *viewer_create(size_t width, size_t height, size_t initial_stride)
{
    if (!ensure_sdl_ready())
    {
        return nullptr;
    }
    ViewerHandle *viewer = new (std::nothrow) ViewerHandle();
    if (!viewer)
    {
        return nullptr;
    }
    viewer->width = width;
    viewer->height = height;
    viewer->stride = initial_stride ? initial_stride : 500;
    viewer->exit_requested = false;
    viewer->expr_frame_buffer.resize(width * height * 3, 0);
    viewer->occ_frame_buffer.resize(width * height * 3, 0);
    viewer->display_scale = 2.0f;
    viewer->window_size = 1024;
    viewer->window_offset = 0;
    viewer->expr_window_size = 1024;
    viewer->expr_window_offset = 0;

    viewer->window = SDL_CreateWindow("Contact-Process RDIM Viewer",
                                      SDL_WINDOWPOS_CENTERED,
                                      SDL_WINDOWPOS_CENTERED,
                                      static_cast<int>(width),
                                      static_cast<int>(height),
                                      SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!viewer->window)
    {
        fprintf(stderr, "SDL_CreateWindow error: %s\n", SDL_GetError());
        delete viewer;
        return nullptr;
    }
    viewer->renderer = SDL_CreateRenderer(viewer->window, -1,
                                          SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!viewer->renderer)
    {
        fprintf(stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError());
        SDL_DestroyWindow(viewer->window);
        delete viewer;
        return nullptr;
    }
    viewer->expr_texture = SDL_CreateTexture(viewer->renderer,
                                             SDL_PIXELFORMAT_RGB24,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             static_cast<int>(width),
                                             static_cast<int>(height));
    viewer->occ_texture = SDL_CreateTexture(viewer->renderer,
                                            SDL_PIXELFORMAT_RGB24,
                                            SDL_TEXTUREACCESS_STREAMING,
                                            static_cast<int>(width),
                                            static_cast<int>(height));
    if (!viewer->expr_texture || !viewer->occ_texture)
    {
        fprintf(stderr, "SDL_CreateTexture error: %s\n", SDL_GetError());
        if (viewer->expr_texture)
        {
            SDL_DestroyTexture(viewer->expr_texture);
        }
        if (viewer->occ_texture)
        {
            SDL_DestroyTexture(viewer->occ_texture);
        }
        SDL_DestroyRenderer(viewer->renderer);
        SDL_DestroyWindow(viewer->window);
        delete viewer;
        return nullptr;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForSDLRenderer(viewer->window, viewer->renderer);
    ImGui_ImplSDLRenderer2_Init(viewer->renderer);

    return viewer;
}

void viewer_destroy(ViewerHandle *viewer)
{
    if (!viewer)
    {
        return;
    }
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyTexture(viewer->expr_texture);
    SDL_DestroyTexture(viewer->occ_texture);
    SDL_DestroyRenderer(viewer->renderer);
    SDL_DestroyWindow(viewer->window);
    delete viewer;
}

void viewer_poll(ViewerHandle *viewer)
{
    if (!viewer)
    {
        return;
    }
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT)
        {
            viewer->exit_requested = true;
        }
        if (event.type == SDL_WINDOWEVENT &&
            event.window.event == SDL_WINDOWEVENT_CLOSE)
        {
            viewer->exit_requested = true;
        }
    }
}

void viewer_submit_frame(ViewerHandle *viewer,
                         const uint8_t *expr_rgb, size_t expr_byte_count,
                         const uint8_t *occ_rgb, size_t occ_byte_count,
                         ViewerFrameMeta meta)
{
    if (!viewer || !expr_rgb || expr_byte_count == 0 || !occ_rgb || occ_byte_count == 0)
    {
        return;
    }
    if (viewer->expr_frame_buffer.size() != expr_byte_count)
    {
        viewer->expr_frame_buffer.resize(expr_byte_count);
    }
    if (viewer->occ_frame_buffer.size() != occ_byte_count)
    {
        viewer->occ_frame_buffer.resize(occ_byte_count);
    }
    std::memcpy(viewer->expr_frame_buffer.data(), expr_rgb, expr_byte_count);
    std::memcpy(viewer->occ_frame_buffer.data(), occ_rgb, occ_byte_count);
    viewer->last_meta = meta;
    const size_t available = meta.mag_history_count;
    const size_t max_offset = (available > viewer->window_size) ? (available - viewer->window_size) : 0;
    if (viewer->window_offset > max_offset)
    {
        viewer->window_offset = max_offset;
    }
    SDL_UpdateTexture(viewer->expr_texture, nullptr, viewer->expr_frame_buffer.data(),
                      static_cast<int>(viewer->width) * 3);
    SDL_UpdateTexture(viewer->occ_texture, nullptr, viewer->occ_frame_buffer.data(),
                      static_cast<int>(viewer->width) * 3);
    render_ui(viewer);
}

bool viewer_wants_exit(const ViewerHandle *viewer)
{
    return viewer ? viewer->exit_requested : false;
}

size_t viewer_current_stride(const ViewerHandle *viewer)
{
    if (!viewer)
    {
        return 1;
    }
    return viewer->stride == 0 ? 1 : viewer->stride;
}

#endif // WITH_GUI
