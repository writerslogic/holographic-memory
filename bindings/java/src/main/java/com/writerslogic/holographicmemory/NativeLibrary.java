// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Finds and loads the native library for this platform.
 *
 * <p>Resolution order: the {@code hms.library.path} system property, the {@code HMS_LIBRARY_PATH}
 * environment variable (each a library file or a directory holding it), a copy bundled in the jar
 * under {@code META-INF/native/<rid>/}, and finally the operating system's library search path.
 */
final class NativeLibrary {
    private NativeLibrary() {}

    /** The .NET-style runtime identifier of this platform: {@code osx-arm64}, {@code linux-x64}, ... */
    static String runtimeId() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String osId;
        if (os.contains("mac") || os.contains("darwin")) {
            osId = "osx";
        } else if (os.contains("win")) {
            osId = "win";
        } else if (os.contains("linux")) {
            osId = "linux";
        } else {
            throw new UnsupportedOperationException("unsupported operating system: " + os);
        }
        String archId;
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            archId = "arm64";
        } else if (arch.equals("amd64") || arch.equals("x86_64")) {
            archId = "x64";
        } else {
            throw new UnsupportedOperationException("unsupported CPU architecture: " + arch);
        }
        return osId + "-" + archId;
    }

    /** The platform file name of the library, e.g. {@code libholographic_memory_c.dylib}. */
    static String fileName() {
        return System.mapLibraryName("holographic_memory_c");
    }

    static SymbolLookup load(Arena arena) {
        String name = fileName();
        for (String candidate : new String[] {System.getProperty("hms.library.path"), System.getenv("HMS_LIBRARY_PATH")}) {
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            Path path = Path.of(candidate);
            if (Files.isDirectory(path)) {
                path = path.resolve(name);
            }
            if (!Files.isRegularFile(path)) {
                throw new IllegalStateException("native library not found at " + path);
            }
            return SymbolLookup.libraryLookup(path, arena);
        }
        String resource = "/META-INF/native/" + runtimeId() + "/" + name;
        try (InputStream in = NativeLibrary.class.getResourceAsStream(resource)) {
            if (in != null) {
                Path dir = Files.createTempDirectory("hms-native-");
                Path file = dir.resolve(name);
                Files.copy(in, file, StandardCopyOption.REPLACE_EXISTING);
                file.toFile().deleteOnExit();
                dir.toFile().deleteOnExit();
                return SymbolLookup.libraryLookup(file, arena);
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot extract bundled native library " + resource, e);
        }
        try {
            return SymbolLookup.libraryLookup(name, arena);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(
                    "native library " + name + " not found: set -Dhms.library.path or HMS_LIBRARY_PATH,"
                            + " bundle META-INF/native/" + runtimeId() + "/" + name
                            + ", or put it on the library search path", e);
        }
    }
}
