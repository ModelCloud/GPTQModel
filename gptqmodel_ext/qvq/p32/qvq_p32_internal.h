// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace qvq_p32_internal {

extern thread_local char last_error[256];

void set_last_error(const char* message);

}  // namespace qvq_p32_internal
