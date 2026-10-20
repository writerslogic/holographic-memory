// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

plugins {
    id("org.jetbrains.kotlin.jvm")
    `maven-publish`
}

group = "com.writerslogic"
version = "0.1.0"

kotlin { jvmToolchain(17) }

dependencies {
    api("net.java.dev.jna:jna:5.19.1")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
}

// scripts/build-jvm-native.sh stages the host library under build/natives/<jna-platform>/.
sourceSets.main { resources.srcDir(layout.buildDirectory.dir("natives")) }

tasks.test {
    useJUnitPlatform()
    // Fail loudly instead of skipping when the native library was not staged.
    doFirst {
        check(layout.buildDirectory.dir("natives").get().asFile.walkTopDown().any { it.isFile }) {
            "no native library staged: run scripts/build-jvm-native.sh first"
        }
    }
}

publishing { publications { create<MavenPublication>("jvm") { from(components["java"]) } } }
