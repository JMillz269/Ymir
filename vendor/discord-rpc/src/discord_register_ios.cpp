// iOS has no Discord IPC socket and no way to register arbitrary launch commands, so registration is a no-op.
#include "discord_register.h"

extern "C" DISCORD_EXPORT void Discord_Register(const char* /*applicationId*/, const char* /*command*/) {}

extern "C" DISCORD_EXPORT void Discord_RegisterSteamGame(const char* /*applicationId*/, const char* /*steamId*/) {}
