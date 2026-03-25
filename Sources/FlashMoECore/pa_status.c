#include "pa_status.h"

const char *pa_status_string(PA_Status status) {
    switch (status) {
        case PA_STATUS_OK:                  return "OK";
        case PA_STATUS_ERROR_GENERIC:       return "Generic error";
        case PA_STATUS_ERROR_OOM:           return "Out of memory";
        case PA_STATUS_ERROR_IO:            return "I/O error";
        case PA_STATUS_ERROR_INVALID_MODEL: return "Invalid model";
        case PA_STATUS_ERROR_METAL_INIT:    return "Metal initialization failed";
        case PA_STATUS_ERROR_LOAD_FAILED:   return "Model load failed";
        case PA_STATUS_CONTEXT_EXHAUSTED:   return "Context exhausted";
        case PA_STATUS_CANCELLED:           return "Cancelled";
        case PA_STATUS_THROTTLED:           return "Throttled";
        default:                            return "Unknown status";
    }
}
