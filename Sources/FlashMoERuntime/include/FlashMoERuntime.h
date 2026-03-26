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

#ifdef __cplusplus
}
#endif

#endif
