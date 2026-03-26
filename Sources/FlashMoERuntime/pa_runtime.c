#include "FlashMoERuntime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

struct PA_Session {
    PA_SessionState state;
    PA_ModelDesc model_desc;
    PA_MemoryBudget memory_budget;
    int model_loaded;
    char last_error[512];
    int cancelled;
    int32_t turn_count;
    PA_GenerationStats last_gen_stats;
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

    session->model_loaded = 1;
    session->state = PA_SESSION_DONE;
    snprintf(session->last_error, sizeof(session->last_error), "No error");
    return PA_STATUS_OK;
}

void pa_session_unload_model(PA_Session *session) {
    if (!session) return;
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

int pa_session_generate(PA_Session *session, const char *prompt,
                        const PA_GenerationConfig *config,
                        PA_TokenCallback callback, void *user_data) {
    if (!session || !prompt) return PA_STATUS_ERROR_GENERIC;

    session->cancelled = 0;

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
        if (session->cancelled) {
            session->state = PA_SESSION_CANCELLED;
            return PA_STATUS_OK;
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
                session->cancelled = 1;
                session->state = PA_SESSION_CANCELLED;
                return PA_STATUS_OK;
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
}

int pa_session_generate_continuation(PA_Session *session, const char *user_message,
                                     const PA_GenerationConfig *config,
                                     PA_TokenCallback callback, void *user_data) {
    return pa_session_generate(session, user_message, config, callback, user_data);
}

void pa_session_cancel(PA_Session *session) {
    if (!session) return;
    session->cancelled = 1;
}

void pa_session_reset(PA_Session *session) {
    if (!session) return;
    session->turn_count = 0;
    memset(&session->last_gen_stats, 0, sizeof(PA_GenerationStats));
    session->cancelled = 0;
    session->state = PA_SESSION_IDLE;
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
