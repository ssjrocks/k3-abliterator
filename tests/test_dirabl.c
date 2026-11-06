/* Unit test for k3_dirabl.h: cc -O2 -o build/test_dirabl tests/test_dirabl.c -lm && build/test_dirabl build/t */
#include "../src/k3_abliterator/engine/k3_dirabl.h"
#include <assert.h>
#include <unistd.h>

#define D 64
static char ctl[600], dirf[600], capf[600];

static void write_ctl(const char *body){ FILE *f=fopen(ctl,"w"); fputs(body,f); fclose(f); }
static void write_dirs(const float *v, int k){ FILE *f=fopen(dirf,"wb"); fwrite(v,4,(size_t)k*D,f); fclose(f); }
static double dot(const float *a, const float *b){ double s=0; for(int d=0;d<D;d++) s+=(double)a[d]*b[d]; return s; }

int main(int argc, char **argv){
    const char *dir = argc>1 ? argv[1] : ".";
    snprintf(ctl,sizeof ctl,"%s/ctl.txt",dir); snprintf(dirf,sizeof dirf,"%s/dir.f32",dir);
    snprintf(capf,sizeof capf,"%s/cap.f32",dir);
    unlink(ctl); unlink(capf);
    setenv("K3_ABL_CTL",ctl,1); dabl_init();

    /* no control file yet: inert */
    assert(dabl_request(D,8,4)==0 && !g_dabl.on);

    float v[2*D]; for(int d=0;d<D;d++){ v[d]=(float)(d%7)-3.f; v[D+d]=(float)(d%5)+v[d]*0.5f; }
    write_dirs(v,2);
    char body[2000];
    snprintf(body,sizeof body,"direction=%s\nscale=1\nlayers=2-5\nembed=0\n",dirf);
    write_ctl(body);
    assert(dabl_request(D,8,4)==DABL_RESET && g_dabl.on && g_dabl.k==2);
    assert(fabs(dot(g_dabl.dir,g_dabl.dir)-1)<1e-5 && fabs(dot(g_dabl.dir,g_dabl.dir+D))<1e-5);
    /* same settings again: no reset */
    assert(dabl_request(D,8,4)==0);

    float x[D], x0[D]; for(int d=0;d<D;d++) x0[d]=x[d]=sinf((float)d);
    dabl_rows(x,1,D,1);                 /* outside the layer range: untouched */
    assert(!memcmp(x,x0,sizeof x));
    dabl_rows(x,1,D,-1);                /* embed=0: untouched */
    assert(!memcmp(x,x0,sizeof x));
    dabl_rows(x,1,D,3);                 /* inside: both directions gone, the rest kept */
    assert(fabs(dot(x,g_dabl.dir))<1e-5 && fabs(dot(x,g_dabl.dir+D))<1e-5);
    float y[D]; memcpy(y,x,sizeof y); dabl_rows(y,1,D,3);
    for(int d=0;d<D;d++) assert(fabs(y[d]-x[d])<1e-5);   /* idempotent at scale 1 */

    /* half strength leaves half the component */
    snprintf(body,sizeof body,"direction=%s\nscale=0.5\nlayers=0-7\n",dirf);
    write_ctl(body);
    assert(dabl_request(D,8,4)==DABL_RESET && g_dabl.scale==0.5f && g_dabl.embed==1);
    memcpy(x,x0,sizeof x); double before=dot(x,g_dabl.dir);
    dabl_rows(x,1,D,-1);
    assert(fabs(dot(x,g_dabl.dir)-0.5*before)<1e-4);

    /* capture: only position 4 is kept; the file appears after finish */
    snprintf(body,sizeof body,"direction=%s\nscale=0.5\nlayers=0-7\ncapture=%s\n",dirf,capf);
    write_ctl(body);
    assert(dabl_request(D,8,4)==0);     /* capture alone doesn't drop state */
    float h[D]; for(int L=0;L<=8;L++){ for(int d=0;d<D;d++) h[d]=(float)(L*100+d);
        dabl_cap(L,3,x0,D); dabl_cap(L,4,h,D); }
    assert(access(capf,F_OK)!=0);
    dabl_cap_finish(1);
    FILE *f=fopen(capf,"rb"); assert(f); float c[9*D]; assert(fread(c,4,9*D,f)==9*D); fclose(f);
    for(int L=0;L<=8;L++) assert(c[L*D+5]==(float)(L*100+5));

    /* rewriting the direction file (same text) is a change */
    v[0]+=1; write_dirs(v,2);
    assert(dabl_request(D,8,4)==DABL_RESET);

    /* scale 0 = off, and turning off drops state once */
    snprintf(body,sizeof body,"direction=%s\nscale=0\n",dirf); write_ctl(body);
    assert(dabl_request(D,8,4)==DABL_RESET && !g_dabl.on);
    assert(dabl_request(D,8,4)==0);

    /* a wrong-sized direction file is an error, not a silent off */
    f=fopen(dirf,"wb"); fwrite(v,4,D-1,f); fclose(f);
    snprintf(body,sizeof body,"direction=%s\n",dirf); write_ctl(body);
    assert(dabl_request(D,8,4)<0 && strstr(g_dabl.err,"want"));

    puts("k3_dirabl: all tests passed");
    return 0;
}
