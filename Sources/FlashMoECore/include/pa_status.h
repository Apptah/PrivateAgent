#ifndef PA_STATUS_H
#define PA_STATUS_H

typedef enum {
    PA_STATUS_OK = 0,
    PA_STATUS_ERROR_GENERIC = -1,
    PA_STATUS_ERROR_OOM = -2,
    PA_STATUS_ERROR_IO = -3,
    PA_STATUS_ERROR_INVALID_MODEL = -4,
    PA_STATUS_ERROR_METAL_INIT = -5,
    PA_STATUS_ERROR_LOAD_FAILED = -6,
    PA_STATUS_CONTEXT_EXHAUSTED = -10,
    PA_STATUS_CANCELLED = -11,
    PA_STATUS_THROTTLED = -12,
} PA_Status;

/// Human-readable string for a status code. Never returns NULL.
const char *pa_status_string(PA_Status status);

#endif // PA_STATUS_H
