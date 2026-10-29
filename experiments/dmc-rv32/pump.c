/* SPDX-License-Identifier: MIT */
#include <stdint.h>
#include "lcd.h"
uint16_t pump_lut[256];
volatile uint32_t qp_strip_ptr,qp_strip_x,qp_strip_y,qp_strip_h,qp_strip_n,qp_strip_flags,qp_strip_serial;
static unsigned xpos,ypos,height,flags;
void pump_begin(int x0,int x1,int y0,int y1,int mode){(void)x1;xpos=x0;ypos=y0;height=y1-y0+1;flags=mode;}
void pump_submit(const uint8_t *cols,int ncols){
 qp_strip_ptr=(uintptr_t)cols;qp_strip_x=xpos;qp_strip_y=ypos;qp_strip_h=height;qp_strip_n=ncols;qp_strip_flags=flags;qp_strip_serial++;
 register long a0 __asm__("a0")=256;register long a7 __asm__("a7")=1;
 __asm__ volatile("ecall":"+r"(a0):"r"(a7):"memory");
 xpos+=ncols*((flags&PUMP_DOUBLE)?2:1);
}
void pump_sync(int n){(void)n;}
int pump_pending(void){return 0;}
void pump_end(void){}
