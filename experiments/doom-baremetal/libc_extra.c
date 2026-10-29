/* SPDX-License-Identifier: MIT */
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
int *__errno_location(void){static int e;return &e;}
void *memset(void*p,int c,size_t n){unsigned char*b=p;while(n--)*b++=c;return p;}
void *memcpy(void*d,const void*s,size_t n){unsigned char*a=d;const unsigned char*b=s;while(n--)*a++=*b++;return d;}
void *memmove(void*d,const void*s,size_t n){unsigned char*a=d;const unsigned char*b=s;if(a<b)return memcpy(d,s,n);while(n){n--;a[n]=b[n];}return d;}
size_t strlen(const char*s){const char*p=s;while(*p)p++;return p-s;}
int strcmp(const char*a,const char*b){while(*a&&*a==*b){a++;b++;}return(unsigned char)*a-(unsigned char)*b;}
int strncmp(const char*a,const char*b,size_t n){while(n&&*a&&*a==*b){a++;b++;n--;}return n?(unsigned char)*a-(unsigned char)*b:0;}
char*strncpy(char*d,const char*s,size_t n){char*r=d;while(n&&*s){*d++=*s++;n--;}while(n--)*d++=0;return r;}
char*strchr(const char*s,int c){do{if(*s==(char)c)return(char*)s;}while(*s++);return NULL;}
char*strrchr(const char*s,int c){const char*r=NULL;do{if(*s==(char)c)r=s;}while(*s++);return(char*)r;}
char*strstr(const char*s,const char*t){size_t n=strlen(t);do{if(!strncmp(s,t,n))return(char*)s;}while(*s++);return NULL;}
char*strdup(const char*s){size_t n=strlen(s)+1;char*p=malloc(n);return memcpy(p,s,n);}
int toupper(int c){return c>='a'&&c<='z'?c-32:c;}
int strncasecmp(const char*a,const char*b,size_t n){while(n&&*a&&toupper(*a)==toupper(*b)){a++;b++;n--;}return n?toupper((unsigned char)*a)-toupper((unsigned char)*b):0;}
int strcasecmp(const char*a,const char*b){return strncasecmp(a,b,SIZE_MAX);}
int abs(int n){return n<0?-n:n;}
int atoi(const char*s){while(*s==' '||*s=='\t')s++;int sign=1,n=0;if(*s=='-'){sign=-1;s++;}else if(*s=='+')s++;while(*s>='0'&&*s<='9')n=n*10+(*s++-'0');return n*sign;}
double atof(const char*s){/* Configuration files are absent on this platform. */return (double)atoi(s);}
double fabs(double x){union{double d;uint64_t i;}v={x};v.i&=UINT64_MAX>>1;return v.d;}
/* Signed division helper avoids a compiler-runtime dependency. */
long long __divdi3(long long a,long long b){int neg=(a<0)^(b<0);unsigned long long x=a<0?-(unsigned long long)a:a,y=b<0?-(unsigned long long)b:b,q=0,r=0;for(int i=63;i>=0;i--){r=(r<<1)|((x>>i)&1);if(r>=y){r-=y;q|=1ULL<<i;}}return neg?-(long long)q:q;}
static void emit(char*b,size_t cap,size_t*pos,char c){if(*pos+1<cap)b[*pos]=c;(*pos)++;}
int vsnprintf(char*b,size_t cap,const char*f,va_list ap){size_t pos=0;while(*f){if(*f!='%'){emit(b,cap,&pos,*f++);continue;}f++;if(*f=='%'){emit(b,cap,&pos,*f++);continue;}int left=0,zero=0,width=0,prec=-1;while(*f=='-'||*f=='0'||*f=='+'||*f==' '||*f=='#'){if(*f=='-')left=1;if(*f=='0')zero=1;f++;}if(*f=='*'){width=va_arg(ap,int);f++;}else while(*f>='0'&&*f<='9')width=width*10+(*f++-'0');if(*f=='.'){f++;prec=0;if(*f=='*'){prec=va_arg(ap,int);f++;}else while(*f>='0'&&*f<='9')prec=prec*10+(*f++-'0');}int length=0;while(*f=='l'||*f=='z'||*f=='h'){if(*f=='l')length++;if(*f=='z')length=1;f++;}char c=*f?*f++:0;char tmp[40];const char*s=tmp;size_t n=0;if(c=='s'){s=va_arg(ap,char*);if(!s)s="(null)";n=strlen(s);if(prec>=0&&n>(size_t)prec)n=prec;}else if(c=='c'){tmp[0]=(char)va_arg(ap,int);n=1;}else if(c=='d'||c=='i'||c=='u'||c=='x'||c=='X'||c=='p'||c=='o'){unsigned v=va_arg(ap,unsigned);unsigned base=c=='x'||c=='X'||c=='p'?16:c=='o'?8:10;int sign=(c=='d'||c=='i')&&(int)v<0;if(sign)v=0-v;char rev[32];size_t k=0;do{unsigned d=v%base;rev[k++]=d<10?'0'+d:(c=='X'?'A':'a')+d-10;v/=base;}while(v);if(sign)tmp[n++]='-';while(prec>(int)k){tmp[n++]='0';prec--;}while(k)tmp[n++]=rev[--k];}else if(c=='f'){(void)va_arg(ap,double);s="0.0";n=3;}else{tmp[0]='?';n=1;}(void)length;int pad=width>(int)n?width-(int)n:0;if(!left)while(pad--)emit(b,cap,&pos,zero?'0':' ');for(size_t i=0;i<n;i++)emit(b,cap,&pos,s[i]);if(left)while(pad-->0)emit(b,cap,&pos,' ');}if(cap)b[pos<cap?pos:cap-1]=0;return pos;}
int snprintf(char*b,size_t n,const char*f,...){va_list ap;va_start(ap,f);int r=vsnprintf(b,n,f,ap);va_end(ap);return r;}
int sscanf(const char*s,const char*f,...){va_list ap;va_start(ap,f);int count=0;while(*f){if(*f==' '||*f=='\t'){while(*s==' '||*s=='\t'||*s=='\n')s++;f++;continue;}if(*f!='%'){if(*s++!=*f++)break;continue;}f++;unsigned base=10;if(*f=='x'||*f=='X')base=16;if(*f=='o')base=8;if(*f!='d'&&*f!='i'&&*f!='u'&&*f!='x'&&*f!='X'&&*f!='o')break;f++;while(*s==' '||*s=='\t')s++;int sign=1;if(*s=='-'){sign=-1;s++;}else if(*s=='+')s++;unsigned v=0,digits=0;while(*s){unsigned d=*s>='0'&&*s<='9'?*s-'0':*s>='a'&&*s<='f'?*s-'a'+10:*s>='A'&&*s<='F'?*s-'A'+10:99;if(d>=base)break;v=v*base+d;s++;digits++;}if(!digits)break;*va_arg(ap,int*)=sign*v;count++;}va_end(ap);return count;}
