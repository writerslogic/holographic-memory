# JNA reflects over these.
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-keep class com.writerslogic.warm.** { *; }
-dontwarn java.awt.*
