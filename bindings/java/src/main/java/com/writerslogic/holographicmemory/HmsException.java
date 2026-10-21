// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

/** A failure reported by the native library. */
public final class HmsException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    /** Native status codes, mirroring the {@code HMS_ERR_*} constants of the C header. */
    public enum Code {
        NULL_POINTER(1), INVALID_ARGUMENT(2), INVALID_UTF8(3), INVALID_HANDLE(4), IO(5), LOCKED(6),
        ENCRYPTION(7), CORRUPT(8), LIMIT_EXCEEDED(9), ABI_MISMATCH(10), PANIC(98), INTERNAL(99),
        /** A code this binding does not know; see {@link HmsException#nativeCode()}. */
        UNKNOWN(-1);

        private final int value;

        Code(int value) {
            this.value = value;
        }

        static Code of(int value) {
            for (Code c : values()) {
                if (c.value == value) {
                    return c;
                }
            }
            return UNKNOWN;
        }
    }

    private final int nativeCode;

    HmsException(int nativeCode, String message) {
        super(message);
        this.nativeCode = nativeCode;
    }

    /** The classified status code. */
    public Code code() {
        return Code.of(nativeCode);
    }

    /** The raw native status code. */
    public int nativeCode() {
        return nativeCode;
    }
}
