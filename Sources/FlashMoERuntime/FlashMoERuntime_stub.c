#include "FlashMoERuntime.h"
#include <stdlib.h>
#include <string.h>
// FlashMoERuntime placeholder. Session logic added in Plan 3.

struct PA_Session {
    PA_MemoryBudget budget;
    char last_error[512];
    int has_model;
};

PA_Session *pa_session_create(void) {
    PA_Session *s = (PA_Session *)calloc(1, sizeof(PA_Session));
    if (s) {
        strncpy(s->last_error, "No error", sizeof(s->last_error) - 1);
    }
    return s;
}

void pa_session_destroy(PA_Session *session) {
    free(session);
}

int pa_session_load_model(PA_Session *session, PA_ModelDesc *desc, uint64_t available_memory_bytes) {
    if (!session || !desc) return PA_STATUS_ERROR_LOAD_FAILED;
    // Stub: compute budget and mark as loaded.
    int result = pa_compute_memory_budget(desc, available_memory_bytes, &session->budget);
    if (result != PA_STATUS_OK) {
        strncpy(session->last_error, pa_status_string((PA_Status)result), sizeof(session->last_error) - 1);
        session->last_error[sizeof(session->last_error) - 1] = '\0';
        return result;
    }
    session->has_model = 1;
    strncpy(session->last_error, "No error", sizeof(session->last_error) - 1);
    return PA_STATUS_OK;
}

void pa_session_unload_model(PA_Session *session) {
    if (!session) return;
    memset(&session->budget, 0, sizeof(session->budget));
    session->has_model = 0;
}

void pa_session_get_memory_budget(PA_Session *session, PA_MemoryBudget *out_budget) {
    if (!session || !out_budget) return;
    *out_budget = session->budget;
}

const char *pa_session_last_error(PA_Session *session) {
    if (!session) return "No session";
    return session->last_error;
}
