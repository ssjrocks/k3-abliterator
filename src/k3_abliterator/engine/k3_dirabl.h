#ifndef K3_DIRABL_H
#define K3_DIRABL_H
/* ============================================================================
 * k3_dirabl.h — directional ablation ("abliteration") for the Kimi K3 engine.
 *
 * OFF unless K3_ABL_CTL names a control file; every dabl_*() below is then
 * inert and the forward is identical to the stock engine.
 *
 * WHY THIS IS EXACT ON K3.  AttnRes replaces the residual stream, but every
 * hidden state the layers read is still a softmax mix of block sums, and every
 * block sum is a plain sum of the embedding and the attention / MLP outputs.
 * Removing a direction r from each of those writers therefore removes it from
 * every mix, and x -= s (x·r) r on a writer's output is the same as baking
 * W -= s r rᵀ W into that writer's output matrix (o_proj, lat_up + shared
 * down_proj, dense down_proj, embed_tokens).
 *
 * CONTROL FILE (re-read at the start of every request, key=value lines):
 *   direction=/path/dirs.f32   k rows of hidden f32 (k>=1); orthonormalized here
 *   scale=1.0                  0 = off, 1 = full removal, >1 overshoots
 *   layers=0-92                inclusive layer range whose writers are ablated
 *   embed=1                    also ablate the embedding rows
 *   capture=/path/cap.f32      write the layer-input mix at the LAST prompt token,
 *                              n_layers+1 rows (the last is the head's input),
 *                              atomically (tmp + rename) after prefill
 * A change to anything but `capture` (or to the direction file itself) makes
 * dabl_request() ask the caller to drop cached state: KV prefixes and recurrent
 * checkpoints computed under the old ablation are not valid under the new one.
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#define DABL_MAXK 16
#define DABL_RESET 1

typedef struct {
    int armed;               /* K3_ABL_CTL set                                  */
    int on;                  /* a direction is loaded and scale != 0            */
    int D, k, lo, hi, embed;
    float scale;
    float *dir;              /* [k][D], orthonormal                             */
    char sig[4096];          /* last control text minus capture + dir file stat */
    int cap_pos;             /* absolute position to capture, -1 = none         */
    int n_rows;
    float *cap;              /* [n_rows][D]                                     */
    char cap_path[1200];
    char err[256];
} K3Dabl;

static K3Dabl g_dabl = { .cap_pos = -1 };

static void dabl_init(void){
    const char *e = getenv("K3_ABL_CTL");
    g_dabl.armed = e && *e;
    if(g_dabl.armed) fprintf(stderr,"[ABL] control file %s\n",e);
}

static void dabl_trim(char *s){
    size_t n = strlen(s);
    while(n && (s[n-1]=='\n'||s[n-1]=='\r'||s[n-1]==' '||s[n-1]=='\t')) s[--n] = 0;
}

/* k rows of D floats -> orthonormal rows (Gram-Schmidt); returns the rank kept. */
static int dabl_orthonormalize(float *v, int k, int D){
    int kept = 0;
    for(int i=0;i<k;i++){
        float *r = v+(size_t)i*D;
        for(int j=0;j<kept;j++){
            const float *q = v+(size_t)j*D; double dot = 0;
            for(int d=0;d<D;d++) dot += (double)r[d]*q[d];
            for(int d=0;d<D;d++) r[d] -= (float)dot*q[d];
        }
        double nn = 0; for(int d=0;d<D;d++) nn += (double)r[d]*r[d];
        if(!(nn > 1e-12)) continue;                 /* dependent or NaN row: drop */
        float inv = (float)(1.0/sqrt(nn));
        float *dst = v+(size_t)kept*D;
        for(int d=0;d<D;d++) dst[d] = r[d]*inv;
        kept++;
    }
    return kept;
}

static int dabl_load_dir(const char *path, int D){
    FILE *f = fopen(path,"rb");
    if(!f){ snprintf(g_dabl.err,sizeof g_dabl.err,"ABL: cannot open direction %s",path); return -1; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    long row = (long)D*4;
    if(n<=0 || n%row || n/row > DABL_MAXK){
        fclose(f);
        snprintf(g_dabl.err,sizeof g_dabl.err,"ABL: %s is %ld bytes, want 1..%d rows of %d f32",
                 path,n,DABL_MAXK,D);
        return -1;
    }
    int k = (int)(n/row);
    float *v = malloc((size_t)n);
    if(!v || fread(v,1,(size_t)n,f)!=(size_t)n){
        fclose(f); free(v);
        snprintf(g_dabl.err,sizeof g_dabl.err,"ABL: short read on %s",path); return -1;
    }
    fclose(f);
    k = dabl_orthonormalize(v,k,D);
    if(k<1){ free(v); snprintf(g_dabl.err,sizeof g_dabl.err,"ABL: %s has no usable direction",path); return -1; }
    free(g_dabl.dir); g_dabl.dir = v; g_dabl.k = k;
    return 0;
}

/* Request start: re-read the control file. last_pos is the last prompt
 * position (the capture target). Returns <0 on error (g_dabl.err says why),
 * else DABL_RESET when cached state must be dropped. */
static int dabl_request(int D, int n_layers, int last_pos){
    g_dabl.cap_pos = -1;
    if(!g_dabl.armed) return 0;
    g_dabl.D = D;
    FILE *f = fopen(getenv("K3_ABL_CTL"),"r");
    char dirp[1200]="", capp[1200]="", sig[4096]="", ln[1200];
    float scale = 1.f; int lo = 0, hi = n_layers-1, embed = 1;
    if(f){
        while(fgets(ln,sizeof ln,f)){
            dabl_trim(ln);
            if(!ln[0] || ln[0]=='#') continue;
            if(!strncmp(ln,"capture=",8)){ snprintf(capp,sizeof capp,"%s",ln+8); continue; }
            if(strlen(sig)+strlen(ln)+2 < sizeof sig){ strcat(sig,ln); strcat(sig,"\n"); }
            if(!strncmp(ln,"direction=",10)) snprintf(dirp,sizeof dirp,"%s",ln+10);
            else if(!strncmp(ln,"scale=",6)) scale = (float)atof(ln+6);
            else if(!strncmp(ln,"layers=",7)){ if(sscanf(ln+7,"%d-%d",&lo,&hi)==1) hi = lo; }
            else if(!strncmp(ln,"embed=",6)) embed = atoi(ln+6);
        }
        fclose(f);
    }
    struct stat st;
    if(dirp[0] && stat(dirp,&st)==0){
        char tail[96]; snprintf(tail,sizeof tail,"@%lld:%lld.%09ld\n",(long long)st.st_size,
                                (long long)st.st_mtim.tv_sec,(long)st.st_mtim.tv_nsec);
        if(strlen(sig)+strlen(tail) < sizeof sig) strcat(sig,tail);
    }
    int ret = 0;
    if(strcmp(sig,g_dabl.sig)){
        int was_on = g_dabl.on;
        g_dabl.on = 0;
        if(dirp[0] && scale!=0.f){
            if(dabl_load_dir(dirp,D)<0){ g_dabl.sig[0] = 0; return -1; }
            g_dabl.on = 1;
        }
        if(lo<0) lo = 0;
        if(hi>n_layers-1) hi = n_layers-1;
        g_dabl.scale = scale; g_dabl.lo = lo; g_dabl.hi = hi; g_dabl.embed = embed;
        snprintf(g_dabl.sig,sizeof g_dabl.sig,"%s",sig);
        if(g_dabl.on || was_on) ret = DABL_RESET;
        if(g_dabl.on) fprintf(stderr,"[ABL] on: %d direction(s) from %s, scale %.3f, layers %d-%d, embed %d\n",
                              g_dabl.k,dirp,scale,lo,hi,embed);
        else fprintf(stderr,"[ABL] off\n");
    }
    if(capp[0]){
        int rows = n_layers+1;
        if(g_dabl.n_rows!=rows){ free(g_dabl.cap); g_dabl.cap = calloc((size_t)rows*D,sizeof(float)); g_dabl.n_rows = rows; }
        if(!g_dabl.cap){ snprintf(g_dabl.err,sizeof g_dabl.err,"ABL: out of memory for capture"); return -1; }
        memset(g_dabl.cap,0,(size_t)rows*D*sizeof(float));
        snprintf(g_dabl.cap_path,sizeof g_dabl.cap_path,"%s",capp);
        g_dabl.cap_pos = last_pos;
    }
    return ret;
}

/* x[C][D] -= scale * Σ_j (x·r_j) r_j, for writers of `layer` (-1 = embedding). */
static inline void dabl_rows(float *x, int C, int D, int layer){
    if(!g_dabl.on) return;
    if(layer<0 ? !g_dabl.embed : (layer<g_dabl.lo || layer>g_dabl.hi)) return;
    for(int t=0;t<C;t++){
        float *row = x+(size_t)t*D;
        for(int j=0;j<g_dabl.k;j++){
            const float *r = g_dabl.dir+(size_t)j*D; double dot = 0;
            for(int d=0;d<D;d++) dot += (double)row[d]*r[d];
            float s = g_dabl.scale*(float)dot;
            for(int d=0;d<D;d++) row[d] -= s*r[d];
        }
    }
}

/* Row `layer` (0..n_layers) of the capture, when this is the capture position. */
static inline void dabl_cap(int layer, int pos, const float *h, int D){
    if(g_dabl.cap_pos<0 || pos!=g_dabl.cap_pos || layer>=g_dabl.n_rows) return;
    memcpy(g_dabl.cap+(size_t)layer*D,h,(size_t)D*sizeof(float));
}

/* After prefill: publish the capture (ok=0 drops it, e.g. on cancel). */
static void dabl_cap_finish(int ok){
    if(g_dabl.cap_pos<0) return;
    g_dabl.cap_pos = -1;
    if(!ok) return;
    char tmp[1210]; snprintf(tmp,sizeof tmp,"%s.tmp",g_dabl.cap_path);
    FILE *f = fopen(tmp,"wb");
    size_t n = (size_t)g_dabl.n_rows*g_dabl.D;
    if(!f || fwrite(g_dabl.cap,sizeof(float),n,f)!=n){
        fprintf(stderr,"[ABL] capture write failed: %s\n",tmp);
        if(f) fclose(f);
        return;
    }
    fclose(f);
    if(rename(tmp,g_dabl.cap_path)) perror("[ABL] capture rename");
}

#endif
