#pragma once

// Defines YMIR_PLATFORM_IOS as 1 when compiling for iOS and 0 otherwise.

#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IOS
        #define YMIR_PLATFORM_IOS 1
    #endif
#endif

#ifndef YMIR_PLATFORM_IOS
    #define YMIR_PLATFORM_IOS 0
#endif
