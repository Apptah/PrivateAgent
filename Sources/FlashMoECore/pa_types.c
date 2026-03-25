#include "pa_types.h"
#include "pa_status.h"
#include <string.h>

int pa_model_desc_validate(const PA_ModelDesc *desc) {
    if (!desc) return PA_STATUS_ERROR_GENERIC;
    if (desc->num_layers == 0 || desc->num_layers > PA_MAX_LAYERS)
        return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->hidden_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->vocab_size == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->num_kv_heads == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->head_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->model_dir[0] == '\0') return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->manifest_version == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    return PA_STATUS_OK;
}

int pa_tensor_ref_validate(const PA_TensorRef *ref) {
    if (!ref) return PA_STATUS_ERROR_GENERIC;
    if (ref->rank == 0 || ref->rank > 4) return PA_STATUS_ERROR_GENERIC;
    for (uint32_t i = 0; i < ref->rank; i++) {
        if (ref->shape[i] == 0) return PA_STATUS_ERROR_GENERIC;
    }
    return PA_STATUS_OK;
}

uint32_t pa_model_desc_full_attn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_FULL_ATTN) count++;
    }
    return count;
}

uint32_t pa_model_desc_gdn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_GDN) count++;
    }
    return count;
}
