#ifndef FLASHMOE_RUNTIME_H
#define FLASHMOE_RUNTIME_H

#include "FlashMoECore.h"

// Session state machine, memory planner, expert pager, layer scheduler.
// Implementations added in Plan 3.

/// Opaque runtime session handle.
typedef struct PA_Session PA_Session;

/// Session states.
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

// ── Session lifecycle ──

/// Create a new session. Returns NULL on allocation failure.
PA_Session *pa_session_create(void);

/// Destroy a session and free all resources.
void pa_session_destroy(PA_Session *session);

/// Load a model into the session.
/// Returns PA_STATUS_OK on success, PA_STATUS_ERROR_LOAD_FAILED on failure.
int pa_session_load_model(PA_Session *session, PA_ModelDesc *desc, uint64_t available_memory_bytes);

/// Unload the current model, returning the session to idle state.
void pa_session_unload_model(PA_Session *session);

/// Get the memory budget for the loaded model. Must be called after a successful load.
void pa_session_get_memory_budget(PA_Session *session, PA_MemoryBudget *out_budget);

/// Get a human-readable description of the last error. Never returns NULL.
const char *pa_session_last_error(PA_Session *session);

#endif // FLASHMOE_RUNTIME_H
