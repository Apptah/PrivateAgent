#include "FlashMoERuntime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct PA_Session {
    PA_SessionState state;
    PA_ModelDesc model_desc;
    PA_MemoryBudget memory_budget;
    int model_loaded;
    char last_error[512];
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
