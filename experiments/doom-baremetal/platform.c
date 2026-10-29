/* SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "doomgeneric.h"
static uint32_t ticks;
extern int gamestate, wipegamestate;

volatile uint32_t qwenputer_framebuffer, qwenputer_frame_bytes, qwenputer_frame_ready;
void DG_Init(void) { puts("QWENPUTER_DOOM_INIT"); }
void DG_DrawFrame(void) {
 qwenputer_framebuffer = (uintptr_t)DG_ScreenBuffer;
 qwenputer_frame_bytes = DOOMGENERIC_RESX * DOOMGENERIC_RESY * sizeof(pixel_t);
 qwenputer_frame_ready++;
}
void DG_SleepMs(uint32_t ms) { ticks += ms; }
uint32_t DG_GetTicksMs(void) { return ++ticks; }
int DG_GetKey(int *pressed, unsigned char *key) { (void)pressed; (void)key; return 0; }
void DG_SetWindowTitle(const char *title) {
 (void)title;
 /* Called after the first game tic. Suppress its startup transition. */
 wipegamestate = gamestate;
}
int main(void) {
 char *argv[] = {"doom", "-iwad", "doom1.wad", "-nosound", "-nomusic", "-nogui", "-warp", "1", "1", "-config", "/no-config", 0};
 doomgeneric_Create(11, argv);
 if (!qwenputer_frame_ready) exit(123);
 puts("QWENPUTER_DOOM_FRAME_READY");
 exit(0);
}
