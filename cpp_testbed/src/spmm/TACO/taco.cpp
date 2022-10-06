//
// Created by lwilkinson on 10/5/22.
//


#include "taco.h"

taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
    taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
    t->order         = order;
    t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
    t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
    t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
    t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
    t->csize         = csize;
    for (int32_t i = 0; i < order; i++) {
        t->dimensions[i]    = dimensions[i];
        t->mode_ordering[i] = mode_ordering[i];
        t->mode_types[i]    = mode_types[i];
        switch (t->mode_types[i]) {
            case taco_mode_dense:
                t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
                break;
            case taco_mode_sparse:
                t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
                break;
        }
    }
    return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
    for (int i = 0; i < t->order; i++) {
        free(t->indices[i]);
    }
    free(t->indices);
    free(t->dimensions);
    free(t->mode_ordering);
    free(t->mode_types);
    free(t);
}