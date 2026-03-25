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

#endif // FLASHMOE_RUNTIME_H
