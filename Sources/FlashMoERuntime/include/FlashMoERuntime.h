#ifndef FLASHMOE_RUNTIME_H
#define FLASHMOE_RUNTIME_H

#include "FlashMoECore.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PA_Session PA_Session;

typedef enum {
    PA_SESSION_IDLE = 0,
    PA_SESSION_LOADING = 1,
    PA_SESSION_PREFILL = 2,
    PA_SESSION_DECODE = 3,
    PA_SESSION_DONE = 4,
    PA_SESSION_CANCELLED = 5,
    PA_SESSION_THROTTLED = 6,
    PA_SESSION_RECOVERING_MEMORY = 7,
} PA_SessionState;

PA_Session *pa_session_create(void);
void pa_session_destroy(PA_Session *session);
PA_SessionState pa_session_get_state(const PA_Session *session);
int pa_session_load_model(PA_Session *session, const PA_ModelDesc *desc, uint64_t available_memory);
void pa_session_unload_model(PA_Session *session);
int pa_session_get_memory_budget(const PA_Session *session, PA_MemoryBudget *out_budget);
const char *pa_session_last_error(const PA_Session *session);

/// Token callback. Return 0 to continue, non-zero to cancel.
typedef int (*PA_TokenCallback)(
    const char *token_text,
    int32_t token_id,
    int32_t tokens_generated,
    double tokens_per_second,
    void *user_data
);

typedef struct {
    int32_t max_tokens;
    float temperature;
    float top_p;
    int32_t think_budget;
} PA_GenerationConfig;

typedef struct {
    double tokens_per_second;
    int32_t tokens_generated;
    double total_time_ms;
    double ttft_ms;
    double prefill_ms;
    int32_t prefill_tokens;
    double prefill_tps;
    uint64_t peak_memory_bytes;
} PA_GenerationStats;

int pa_session_generate(PA_Session *session, const char *prompt, const PA_GenerationConfig *config, PA_TokenCallback callback, void *user_data);
int pa_session_generate_continuation(PA_Session *session, const char *user_message, const PA_GenerationConfig *config, PA_TokenCallback callback, void *user_data);
void pa_session_cancel(PA_Session *session);
void pa_session_reset(PA_Session *session);
int pa_session_get_gen_stats(const PA_Session *session, PA_GenerationStats *out_stats);
int32_t pa_session_turn_count(const PA_Session *session);

#ifdef __cplusplus
}
#endif

#endif
