/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
extern unsigned char __heap_start[], __heap_end[], __wad_start[], __wad_end[];
static unsigned char *heap;
static int err;

static long ec(long n,long a) { register long r0 __asm__("a0")=a; register long r7 __asm__("a7")=n; __asm__ volatile("ecall":"+r"(r0):"r"(r7):"memory"); return r0; }
_Noreturn void exit(int status) { ec(93,status); for(;;){} }
_Noreturn void abort(void) { exit(127); }
void *malloc(size_t n) { if(!heap) heap=__heap_start; n=(n+31)&~15u; if(n>(size_t)(__heap_end-heap)) { puts("QWENPUTER_HEAP_EXHAUSTED"); exit(125); } size_t *p=(size_t*)heap; *p=n-16; heap+=n; return (unsigned char*)p+16; }
void free(void *p) { (void)p; }
void *calloc(size_t n,size_t m) { if(m && n>SIZE_MAX/m) return NULL; void *p=malloc(n*m); memset(p,0,n*m); return p; }
/* Engine realloc callers only grow configuration/array buffers. Save allocation sizes. */
void *realloc(void *p,size_t n) { if(!p)return malloc(n); size_t old=((size_t*)((unsigned char*)p-16))[0]; void *q=malloc(n); memcpy(q,p,old<n?old:n); return q; }
int putchar(int c) { ec(1,(unsigned char)c); return (unsigned char)c; }
int puts(const char *s) { while(*s)putchar(*s++);putchar('\n');return 0; }
FILE *const stdin=(FILE*)1; FILE *const stdout=(FILE*)2; FILE *const stderr=(FILE*)3;
static struct {size_t pos; int used;} files[8];
static int fileid(FILE *f) { uintptr_t v=(uintptr_t)f; return v>=16 && v<24 ? (int)v-16:-1; }
FILE *fopen(const char *p,const char *mode) { const char *b=strrchr(p,'/'); b=b?b+1:p; if((mode[0]!='r') || strcasecmp(b,"doom1.wad") || (uintptr_t)__wad_end==(uintptr_t)__wad_start){err=ENOENT;return NULL;} for(int i=0;i<8;i++)if(!files[i].used){files[i].used=1;files[i].pos=0;return (FILE*)(uintptr_t)(i+16);} return NULL; }
int fclose(FILE *f) { int i=fileid(f);if(i>=0)files[i].used=0;return 0; }
int fseek(FILE *f,long offset,int whence) { int i=fileid(f);if(i<0)return -1;long p=whence==SEEK_SET?offset:whence==SEEK_CUR?(long)files[i].pos+offset:(long)(__wad_end-__wad_start)+offset;if(p<0 || p>__wad_end-__wad_start)return -1;files[i].pos=p;return 0; }
long ftell(FILE *f) {int i=fileid(f);return i<0?-1:(long)files[i].pos;}
size_t fread(void *buf,size_t size,size_t n,FILE *f) {int i=fileid(f);if(i<0||!size)return 0;size_t left=(__wad_end-__wad_start)-files[i].pos;if(n>left/size)n=left/size;memcpy(buf,__wad_start+files[i].pos,n*size);files[i].pos+=n*size;return n;}
size_t fwrite(const void *b,size_t s,size_t n,FILE *f){if(fileid(f)>=0)return 0;const char *p=b;for(size_t i=0;i<s*n;i++)putchar(p[i]);return n;}
int fputc(int c,FILE*f){(void)f;return putchar(c);}
int fputs(const char*s,FILE*f){(void)f;while(*s)putchar(*s++);return 0;}
int fflush(FILE*f){(void)f;return 0;}
int remove(const char*p){(void)p;return -1;}
int rename(const char*a,const char*b){(void)a;(void)b;return -1;}
char *getenv(const char*n){(void)n;return NULL;}
int mkdir(const char*p,mode_t m){(void)p;(void)m;return -1;}
int system(const char*s){(void)s;return -1;}
int isatty(int fd){(void)fd;return 1;}
int atexit(void(*f)(void)){(void)f;return 0;}
int fprintf(FILE*f,const char*fmt,...){char b[2048];va_list ap;va_start(ap,fmt);int n=vsnprintf(b,sizeof b,fmt,ap);va_end(ap);fputs(b,f);return n;}
int vfprintf(FILE*f,const char*fmt,va_list ap){char b[2048];int n=vsnprintf(b,sizeof b,fmt,ap);fputs(b,f);return n;}
int printf(const char*fmt,...){va_list ap;va_start(ap,fmt);int n=vfprintf(stdout,fmt,ap);va_end(ap);return n;}
