// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.warm;

/**
 * The store's protection state.
 *
 * @param securityCompiled false when the native library was built without encryption support
 * @param encryptionActive data at rest is encrypted
 * @param signingActive mutations are signed
 * @param auditActive the audit log is on
 * @param differentialPrivacyActive differential privacy is on
 * @param epsilon the privacy budget
 */
public record SecurityStatus(
        boolean securityCompiled,
        boolean encryptionActive,
        boolean signingActive,
        boolean auditActive,
        boolean differentialPrivacyActive,
        double epsilon) {}
