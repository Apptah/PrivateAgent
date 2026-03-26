#include "FlashMoERuntime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>

#ifdef PA_USE_REAL_ENGINE
#include "FlashMoEEngine.h"
#endif

struct PA_Session {
    PA_SessionState state;
    PA_ModelDesc model_desc;
    PA_MemoryBudget memory_budget;
    int model_loaded;
    char last_error[512];
    atomic_int cancelled;
    int32_t turn_count;
    PA_GenerationStats last_gen_stats;
#ifdef PA_USE_REAL_ENGINE
    void *engine_ctx;  // FlashMoEContext* (void* to avoid header dependency issues)
#endif
};

PA_Session *pa_session_create(void) {
    PA_Session *s = calloc(1, sizeof(PA_Session));
    if (!s) return NULL;
    s->state = PA_SESSION_IDLE;
    strcpy(s->last_error, "No error");
    return s;
}

void pa_session_destroy(PA_Session *session) {
    if (!session) return;
#ifdef PA_USE_REAL_ENGINE
    if (session->engine_ctx) {
        FlashMoEContext *ctx = (FlashMoEContext *)session->engine_ctx;
        if (session->model_loaded) {
            flashmoe_unload(ctx);
        }
        flashmoe_destroy(ctx);
        session->engine_ctx = NULL;
    }
#endif
    free(session);
}

PA_SessionState pa_session_get_state(const PA_Session *session) {
    if (!session) return PA_SESSION_IDLE;
    return session->state;
}

int pa_session_load_model(PA_Session *session, const PA_ModelDesc *desc, uint64_t available_memory) {
    if (!session || !desc) {
        if (session) snprintf(session->last_error, sizeof(session->last_error), "NULL argument");
        return PA_STATUS_ERROR_GENERIC;
    }

    session->state = PA_SESSION_LOADING;

    int status = pa_model_desc_validate(desc);
    if (status != PA_STATUS_OK) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Invalid model descriptor: %s", pa_status_string(status));
        session->state = PA_SESSION_IDLE;
        return status;
    }

    memcpy(&session->model_desc, desc, sizeof(PA_ModelDesc));

    status = pa_compute_memory_budget(desc, available_memory, &session->memory_budget);
    if (status != PA_STATUS_OK) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Insufficient memory (available: %llu bytes)", (unsigned long long)available_memory);
        session->state = PA_SESSION_IDLE;
        return status;
    }

#ifdef PA_USE_REAL_ENGINE
    // Destroy any existing engine context
    if (session->engine_ctx) {
        FlashMoEContext *old_ctx = (FlashMoEContext *)session->engine_ctx;
        if (session->model_loaded) {
            flashmoe_unload(old_ctx);
        }
        flashmoe_destroy(old_ctx);
        session->engine_ctx = NULL;
    }

    FlashMoEContext *ctx = flashmoe_create();
    if (!ctx) {
        snprintf(session->last_error, sizeof(session->last_error), "Engine create failed");
        session->state = PA_SESSION_IDLE;
        return PA_STATUS_ERROR_LOAD_FAILED;
    }

    FlashMoEConfig fmConfig = {0};
    fmConfig.model_path = desc->model_dir;
    fmConfig.max_context = (int)session->memory_budget.max_context_length;
    fmConfig.think_budget = 512;  // limit thinking tokens to prevent loops
    fmConfig.temperature = 0.7f;  // Qwen3 recommended for non-thinking
    fmConfig.top_p = 0.8f;
    fmConfig.top_k = 20;
    fmConfig.prefill_batch = 32;           // batch 32 tokens per layer during prefill
    fmConfig.prefill_skip_experts = 1;     // skip routed experts for intermediate prefill tokens
    fmConfig.prefill_experts_full_only = 1; // only load routed experts at full attention layers
    fmConfig.verbose = 0;

    int loadResult = flashmoe_load(ctx, &fmConfig);
    if (loadResult != 0) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Engine load failed: %s", flashmoe_last_error(ctx));
        flashmoe_destroy(ctx);
        session->state = PA_SESSION_IDLE;
        return PA_STATUS_ERROR_LOAD_FAILED;
    }

    session->engine_ctx = ctx;
#endif

    session->model_loaded = 1;
    session->state = PA_SESSION_DONE;
    snprintf(session->last_error, sizeof(session->last_error), "No error");
    return PA_STATUS_OK;
}

void pa_session_unload_model(PA_Session *session) {
    if (!session) return;
#ifdef PA_USE_REAL_ENGINE
    if (session->engine_ctx) {
        FlashMoEContext *ctx = (FlashMoEContext *)session->engine_ctx;
        flashmoe_unload(ctx);
        flashmoe_destroy(ctx);
        session->engine_ctx = NULL;
    }
#endif
    session->model_loaded = 0;
    memset(&session->model_desc, 0, sizeof(PA_ModelDesc));
    memset(&session->memory_budget, 0, sizeof(PA_MemoryBudget));
    session->state = PA_SESSION_IDLE;
}

int pa_session_get_memory_budget(const PA_Session *session, PA_MemoryBudget *out_budget) {
    if (!session || !out_budget) return PA_STATUS_ERROR_GENERIC;
    if (!session->model_loaded) return PA_STATUS_ERROR_GENERIC;
    memcpy(out_budget, &session->memory_budget, sizeof(PA_MemoryBudget));
    return PA_STATUS_OK;
}

const char *pa_session_last_error(const PA_Session *session) {
    if (!session) return "NULL session";
    return session->last_error;
}

static double timespec_to_ms(const struct timespec *ts) {
    return (double)ts->tv_sec * 1000.0 + (double)ts->tv_nsec / 1.0e6;
}

// ---- Callback bridge: PA_TokenCallback → FlashMoETokenCallback ----

struct pa_callback_bridge {
    PA_TokenCallback user_callback;
    void *user_data;
};

#ifdef PA_USE_REAL_ENGINE
static int pa_bridge_callback(const char *text, int id, int generated, double tps, void *ud) {
    struct pa_callback_bridge *bridge = (struct pa_callback_bridge *)ud;
    if (!bridge->user_callback) return 0;
    return bridge->user_callback(text, (int32_t)id, (int32_t)generated, tps, bridge->user_data);
}
#endif

int pa_session_generate(PA_Session *session, const char *prompt,
                        const PA_GenerationConfig *config,
                        PA_TokenCallback callback, void *user_data) {
    if (!session || !prompt) return PA_STATUS_ERROR_GENERIC;

    atomic_store_explicit(&session->cancelled, 0, memory_order_release);

#ifdef PA_USE_REAL_ENGINE
    if (!session->engine_ctx) {
        snprintf(session->last_error, sizeof(session->last_error), "No engine loaded");
        return PA_STATUS_ERROR_GENERIC;
    }

    FlashMoEContext *ctx = (FlashMoEContext *)session->engine_ctx;
    struct pa_callback_bridge bridge = { callback, user_data };

    int max_tokens = config ? (int)config->max_tokens : 512;

    session->state = PA_SESSION_PREFILL;
    int result = flashmoe_generate(ctx, prompt, max_tokens, pa_bridge_callback, &bridge);

    if (atomic_load_explicit(&session->cancelled, memory_order_acquire)) {
        session->state = PA_SESSION_CANCELLED;
        return PA_STATUS_CANCELLED;
    }

    if (result < 0) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Generation failed: %s", flashmoe_last_error(ctx));
        session->state = PA_SESSION_DONE;
        return PA_STATUS_ERROR_GENERIC;
    }

    // Pull stats from engine
    FlashMoEStats fmStats = {0};
    flashmoe_get_stats(ctx, &fmStats);

    session->last_gen_stats.tokens_per_second  = fmStats.tokens_per_second;
    session->last_gen_stats.tokens_generated   = (int32_t)fmStats.tokens_generated;
    session->last_gen_stats.total_time_ms      = fmStats.total_time_ms;
    session->last_gen_stats.ttft_ms            = fmStats.ttft_ms;
    session->last_gen_stats.prefill_ms         = fmStats.prefill_ms;
    session->last_gen_stats.prefill_tokens     = (int32_t)fmStats.prefill_tokens;
    session->last_gen_stats.prefill_tps        = fmStats.prefill_tps;
    session->last_gen_stats.peak_memory_bytes  = (uint64_t)fmStats.metal_buffer_bytes;

    session->turn_count++;
    session->state = PA_SESSION_DONE;
    return PA_STATUS_OK;

#else  // PA_USE_REAL_ENGINE not defined → mock path

    // Mock prefill
    session->state = PA_SESSION_PREFILL;
    struct timespec t_start, t_prefill_end, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    usleep(50000); // 50ms mock prefill
    clock_gettime(CLOCK_MONOTONIC, &t_prefill_end);

    double prefill_ms = timespec_to_ms(&t_prefill_end) - timespec_to_ms(&t_start);

    // Estimate prefill tokens from prompt length (rough: 4 chars per token)
    int32_t prefill_tokens = (int32_t)(strlen(prompt) / 4 + 1);
    double prefill_tps = (prefill_ms > 0.0) ? (prefill_tokens * 1000.0 / prefill_ms) : 0.0;

    session->state = PA_SESSION_DECODE;

    static const char *mock_tokens[] = {
        "Hello", " ", "!", " I", "'m", " Private", "Agent", "."
    };
    int mock_count = (int)(sizeof(mock_tokens) / sizeof(mock_tokens[0]));

    double ttft_ms = 0.0;
    int32_t tokens_generated = 0;

    for (int i = 0; i < mock_count; i++) {
        if (atomic_load_explicit(&session->cancelled, memory_order_acquire)) {
            session->state = PA_SESSION_CANCELLED;
            return PA_STATUS_CANCELLED;
        }

        usleep(10000); // 10ms per token

        tokens_generated++;
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed_ms = timespec_to_ms(&t_now) - timespec_to_ms(&t_prefill_end);

        if (i == 0) {
            ttft_ms = elapsed_ms;
        }

        double tps = (elapsed_ms > 0.0) ? (tokens_generated * 1000.0 / elapsed_ms) : 0.0;

        if (callback) {
            int cancel = callback(mock_tokens[i], (int32_t)i, tokens_generated, tps, user_data);
            if (cancel != 0) {
                atomic_store_explicit(&session->cancelled, 1, memory_order_release);
                session->state = PA_SESSION_CANCELLED;
                return PA_STATUS_CANCELLED;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double total_ms = timespec_to_ms(&t_now) - timespec_to_ms(&t_start);
    double decode_ms = timespec_to_ms(&t_now) - timespec_to_ms(&t_prefill_end);
    double final_tps = (decode_ms > 0.0) ? (tokens_generated * 1000.0 / decode_ms) : 0.0;

    session->last_gen_stats.tokens_per_second = final_tps;
    session->last_gen_stats.tokens_generated = tokens_generated;
    session->last_gen_stats.total_time_ms = total_ms;
    session->last_gen_stats.ttft_ms = ttft_ms;
    session->last_gen_stats.prefill_ms = prefill_ms;
    session->last_gen_stats.prefill_tokens = prefill_tokens;
    session->last_gen_stats.prefill_tps = prefill_tps;
    session->last_gen_stats.peak_memory_bytes = 0;

    session->turn_count++;
    session->state = PA_SESSION_DONE;
    return PA_STATUS_OK;

#endif  // PA_USE_REAL_ENGINE
}

int pa_session_generate_continuation(PA_Session *session, const char *user_message,
                                     const PA_GenerationConfig *config,
                                     PA_TokenCallback callback, void *user_data) {
#ifdef PA_USE_REAL_ENGINE
    if (!session || !user_message) return PA_STATUS_ERROR_GENERIC;
    if (!session->engine_ctx) {
        snprintf(session->last_error, sizeof(session->last_error), "No engine loaded");
        return PA_STATUS_ERROR_GENERIC;
    }

    atomic_store_explicit(&session->cancelled, 0, memory_order_release);

    FlashMoEContext *ctx = (FlashMoEContext *)session->engine_ctx;
    struct pa_callback_bridge bridge = { callback, user_data };

    int max_tokens = config ? (int)config->max_tokens : 512;

    session->state = PA_SESSION_PREFILL;
    int result = flashmoe_generate_continuation(ctx, user_message, max_tokens,
                                                pa_bridge_callback, &bridge);

    if (atomic_load_explicit(&session->cancelled, memory_order_acquire)) {
        session->state = PA_SESSION_CANCELLED;
        return PA_STATUS_CANCELLED;
    }

    if (result < 0) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Continuation failed: %s", flashmoe_last_error(ctx));
        session->state = PA_SESSION_DONE;
        return PA_STATUS_ERROR_GENERIC;
    }

    // Pull stats from engine
    FlashMoEStats fmStats = {0};
    flashmoe_get_stats(ctx, &fmStats);

    session->last_gen_stats.tokens_per_second  = fmStats.tokens_per_second;
    session->last_gen_stats.tokens_generated   = (int32_t)fmStats.tokens_generated;
    session->last_gen_stats.total_time_ms      = fmStats.total_time_ms;
    session->last_gen_stats.ttft_ms            = fmStats.ttft_ms;
    session->last_gen_stats.prefill_ms         = fmStats.prefill_ms;
    session->last_gen_stats.prefill_tokens     = (int32_t)fmStats.prefill_tokens;
    session->last_gen_stats.prefill_tps        = fmStats.prefill_tps;
    session->last_gen_stats.peak_memory_bytes  = (uint64_t)fmStats.metal_buffer_bytes;

    session->turn_count++;
    session->state = PA_SESSION_DONE;
    return PA_STATUS_OK;

#else
    return pa_session_generate(session, user_message, config, callback, user_data);
#endif
}

void pa_session_cancel(PA_Session *session) {
    if (!session) return;
    printf("[C] pa_session_cancel called, state=%d\n", session->state);
    atomic_store_explicit(&session->cancelled, 1, memory_order_release);
#ifdef PA_USE_REAL_ENGINE
    if (session->engine_ctx) {
        flashmoe_cancel((FlashMoEContext *)session->engine_ctx);
    }
#endif
}

void pa_session_reset(PA_Session *session) {
    if (!session) return;
    session->turn_count = 0;
    memset(&session->last_gen_stats, 0, sizeof(PA_GenerationStats));
    atomic_store_explicit(&session->cancelled, 0, memory_order_release);
    session->state = PA_SESSION_IDLE;
#ifdef PA_USE_REAL_ENGINE
    if (session->engine_ctx) {
        flashmoe_reset((FlashMoEContext *)session->engine_ctx);
    }
#endif
}

int pa_session_get_gen_stats(const PA_Session *session, PA_GenerationStats *out_stats) {
    if (!session || !out_stats) return PA_STATUS_ERROR_GENERIC;
    memcpy(out_stats, &session->last_gen_stats, sizeof(PA_GenerationStats));
    return PA_STATUS_OK;
}

int32_t pa_session_turn_count(const PA_Session *session) {
    if (!session) return 0;
    return session->turn_count;
}
