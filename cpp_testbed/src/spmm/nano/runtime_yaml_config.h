//
// Created by lwilkinson on 9/30/22.
//

#pragma once

#include "SpMM_SOP.h"
#include "mapping_to_executor.h"

using CSRTypes = sop::CSRStorageTypes<int, int>;
using NO_PACKING = sop::PackingDesc<sop::NO_PACKING, sop::NO_PACKING>;
using C_PARTIALY_PACKED = sop::PackingDesc<sop::PARTIAL_PACKING, sop::NO_PACKING>;