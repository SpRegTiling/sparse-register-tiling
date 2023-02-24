// temp.grp remove

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
//#include <xmmintrin.h>
//#include "mkl.h"
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include<math.h>
#include<iostream>

#include "assume_aligned.h"
#include "ASpT.h"

using namespace std;

double time_in_mill_now();
double time_in_mill_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double time_in_mill =
    (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
  return time_in_mill;
}

#define ERR fprintf(stderr, "ERR\n");

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE float

constexpr unsigned floorlog2(unsigned x)
{
    return x == 1 ? 0 : 1+floorlog2(x >> 1);
}


#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
#define THRESHOLD (16*1)
#define BH (ASpT_block_height)
#define LOG_BH (floorlog2(BH))
#define BW (128*1)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (-1)
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define _NTHREAD (68)
#define SC_SIZE (2048)

//#define SIM_VALUE

struct v_struct {
	int row, col;
	FTYPE val;
	int grp;
};

//double vari, avg;
//double avg0[NTHREAD];
//struct v_struct *temp_v, *gold_temp_v;
//int sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
//int nr0;

//int *csr_v;
//int *csr_e, *csr_e0;
//FTYPE *csr_ev, *csr_ev0;
////int *mcsr_v;

//int compare0(const void *a, const void *b)
//{
//        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
//        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
//        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
//}
//
//int compare1(const void *a, const void *b)
//{
//        if ((((struct v_struct *)a)->row)/BH - (((struct v_struct *)b)->row)/BH > 0) return 1;
//        if ((((struct v_struct *)a)->row)/BH - (((struct v_struct *)b)->row)/BH < 0) return -1;
//        if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
//        if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
//        return ((struct v_struct *)a)->row - ((struct v_struct *)b)->row;
//}
//
//int compare2(const void *a, const void *b)
//{
//        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
//        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
//        if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
//        if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
//        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
//}


//void ready(int argc, char **argv)
//{
//        FILE *fp;
//        int *loc;
//        char buf[300];
//        int nflag, sflag;
//        int pre_count=0, tmp_ne;
//        int i;
//
//        fprintf(stdout, "TTAAGG,%s,", argv[1]);
//
//        ////sc = atoi(argv[2]);
//        sc=128;
//
//        fp = fopen(argv[1], "r");
//        fgets(buf, 300, fp);
//        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
//        else sflag = 0;
//        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
//        else if(strstr(buf, "complex") != NULL) nflag = -1;
//        else nflag = 1;
//
//#ifdef SYM
//        sflag = 1;
//#endif
//
//        while(1) {
//                pre_count++;
//                fgets(buf, 300, fp);
//                if(strstr(buf, "%") == NULL) break;
//        }
//        fclose(fp);
//
//        fp = fopen(argv[1], "r");
//        for(i=0;i<pre_count;i++)
//                fgets(buf, 300, fp);
//
//        fscanf(fp, "%d %d %d", &nr, &nc, &ne);
//        nr0 = nr;
//        ne *= (sflag+1);
//        nr = CEIL(nr,BH)*BH;
//	    npanel = CEIL(nr,BH);
//
//        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
//        gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
//
//        for(i=0;i<ne;i++) {
//                fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
//		temp_v[i].grp = INIT_GRP;
//                temp_v[i].row--; temp_v[i].col--;
//
//                if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
//                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
//                        exit(0);
//                }
//                if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
//                else if(nflag == 1) {
//                        FTYPE ftemp;
//                        fscanf(fp, " %f ", &ftemp);
//                        temp_v[i].val = ftemp;
//                } else { // complex
//                        FTYPE ftemp1, ftemp2;
//                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
//                        temp_v[i].val = ftemp1;
//                }
//#ifdef SIM_VALUE
//temp_v[i].val = 1.0f;
//#endif
//                if(sflag == 1) {
//                        i++;
//                        temp_v[i].row = temp_v[i-1].col;
//                        temp_v[i].col = temp_v[i-1].row;
//                        temp_v[i].val = temp_v[i-1].val;
//        		temp_v[i].grp = INIT_GRP;
//	        }
//        }
//        qsort(temp_v, ne, sizeof(struct v_struct), compare0);
//
//        loc = (int *)malloc(sizeof(int)*(ne+1));
//
//        memset(loc, 0, sizeof(int)*(ne+1));
//        loc[0]=1;
//        for(i=1;i<ne;i++) {
//                if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
//                        loc[i] = 0;
//                else loc[i] = 1;
//        }
//        for(i=1;i<=ne;i++)
//                loc[i] += loc[i-1];
//        for(i=ne; i>=1; i--)
//                loc[i] = loc[i-1];
//        loc[0] = 0;
//
//        for(i=0;i<ne;i++) {
//                temp_v[loc[i]].row = temp_v[i].row;
//                temp_v[loc[i]].col = temp_v[i].col;
//                temp_v[loc[i]].val = temp_v[i].val;
//                temp_v[loc[i]].grp = temp_v[i].grp;
//        }
//        ne = loc[ne];
//        temp_v[ne].row = nr;
//        gold_ne = ne;
//        for(i=0;i<=ne;i++) {
//                gold_temp_v[i].row = temp_v[i].row;
//                gold_temp_v[i].col = temp_v[i].col;
//                gold_temp_v[i].val = temp_v[i].val;
//                gold_temp_v[i].grp = temp_v[i].grp;
//        }
//        free(loc);
//
//	csr_v = (int *)malloc(sizeof(int)*(nr+1));
//	csr_e0 = (int *)malloc(sizeof(int)*ne);
//	csr_ev0 = (FTYPE *)malloc(sizeof(FTYPE)*ne);
//	memset(csr_v, 0, sizeof(int)*(nr+1));
//
//        for(i=0;i<ne;i++) {
//                csr_e0[i] = temp_v[i].col;
//                csr_ev0[i] = temp_v[i].val;
//                csr_v[1+temp_v[i].row] = i+1;
//        }
//
//        for(i=1;i<nr;i++) {
//                if(csr_v[i] == 0) csr_v[i] = csr_v[i-1];
//        }
//	csr_v[nr] = ne;
//
//	csr_e = (int *)malloc(sizeof(int)*ne);
//	csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*ne);
//
//	fprintf(stdout,"%d,%d,%d,",nr0,nc,ne);
//}


InspectorMetadata<FTYPE> inspect(
    int nr0, int nc, int ne,
    int* row_ptrs, int* col_indices, FTYPE* values,
    int NTHREAD,
    int ASpT_block_height
) {
    int *special = NULL;
    int *special2 = NULL;
    int special_p = NULL;

    char scr_pad[NTHREAD][SC_SIZE];


    int sflag = 0;
    double avg0[NTHREAD];
    double avg = 0;
    double vari = 0;

    special = (int *)malloc(sizeof(int)*ne);
    special2 = (int *)malloc(sizeof(int)*ne);
    memset(special, 0, sizeof(int)*ne);
    memset(special2, 0, sizeof(int)*ne);

    ne *= (sflag+1);
    int nr = CEIL(nr0,BH)*BH;
    int npanel = CEIL(nr,BH);

    int* csr_e0 = col_indices;
    FTYPE* csr_ev0 = values;

    // Pad out row_ptrs
    int* csr_v = (int *)malloc(sizeof(int)*(nr+1));

    int i = 0;
    for (; i < nr0 + 1; i++) csr_v[i] = row_ptrs[i];
    for (; i < nr + 1; i++)  csr_v[i] = row_ptrs[nr0];

    int* csr_e = (int *)malloc(sizeof(int)*ne);
    FTYPE* csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*ne);

    int* mcsr_cnt = (int *)malloc(sizeof(int)*(npanel+1));
    int* mcsr_chk = (int *)malloc(sizeof(int)*(npanel+1));
    int* mcsr_e = (int *)malloc(sizeof(int)*ne); // reduced later

    memset(mcsr_cnt, 0, sizeof(int)*(npanel+1));
    memset(mcsr_chk, 0, sizeof(int)*(npanel+1));
    memset(mcsr_e, 0, sizeof(int)*ne);


    int bv_size = CEIL(nc, 32);
    unsigned int **bv = (unsigned int **)malloc(sizeof(unsigned int *)*NTHREAD);
    for(int i=0;i<NTHREAD;i++)
        bv[i] = (unsigned int *)malloc(sizeof(unsigned int)*bv_size);
    int **csr_e1 = (int **)malloc(sizeof(int *)*2);
    short **coo = (short **)malloc(sizeof(short *)*2);
    for(int i=0;i<2;i++) {
        csr_e1[i] = (int *)malloc(sizeof(int)*ne);
        coo[i] = (short *)malloc(sizeof(short)*ne);
    }

    // filtering(WILL)
    //memcpy(csr_e1[0], csr_e0, sizeof(int)*ne);
#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {
        for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
            for(int j=csr_v[i]; j<csr_v[i+1]; j++) {
                csr_e1[0][j] = csr_e0[j];
            }
        }

    }

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {
        int tid = omp_get_thread_num();
        int i, j, t_sum=0;

        // coo generate and mcsr_chk
        memset(scr_pad[tid], 0, sizeof(char)*SC_SIZE);
        for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
            for(j=csr_v[i]; j<csr_v[i+1]; j++) {
                coo[0][j] = (i&(BH-1));
                int k = (csr_e0[j]&(SC_SIZE-1));
                if(scr_pad[tid][k] < THRESHOLD) {
                    if(scr_pad[tid][k] == THRESHOLD - 1) t_sum++;
                    scr_pad[tid][k]++;
                }
            }
        }

        if (t_sum < MIN_OCC) {
            mcsr_chk[row_panel] = 1;
            mcsr_cnt[row_panel+1] = 1;
            continue;
        }

        // sorting(merge sort)
        int flag = 0;
        for(int stride = 1; stride <= BH/2; stride *= 2, flag=1-flag) {
            for(int pivot = row_panel*BH; pivot < (row_panel+1)*BH; pivot += stride*2) {
                int l1, l2;
                for(i = l1 = csr_v[pivot], l2 = csr_v[pivot+stride]; l1 < csr_v[pivot+stride] && l2 < csr_v[pivot+stride*2]; i++) {
                    if(csr_e1[flag][l1] <= csr_e1[flag][l2]) {
                        coo[1-flag][i] = coo[flag][l1];
                        csr_e1[1-flag][i] = csr_e1[flag][l1++];
                    }
                    else {
                        coo[1-flag][i] = coo[flag][l2];
                        csr_e1[1-flag][i] = csr_e1[flag][l2++];
                    }
                }
                while(l1 < csr_v[pivot+stride]) {
                    coo[1-flag][i] = coo[flag][l1];
                    csr_e1[1-flag][i++] = csr_e1[flag][l1++];
                }
                while(l2 < csr_v[pivot+stride*2]) {
                    coo[1-flag][i] = coo[flag][l2];
                    csr_e1[1-flag][i++] = csr_e1[flag][l2++];
                }
            }
        }

        int weight=1;

        int cq=0, cr=0;

        // dense bit extract (and mcsr_e making)
        for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
            if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
            else {
                if(weight >= THRESHOLD) {
                    cr++;
                } 				//if(cr == BW) { cq++; cr=0;}
                weight = 1;
            }
        }
        //int reminder = (csr_e1[flag][i-1]&31);
        if(weight >= THRESHOLD) {
            cr++;
        } 		//if(cr == BW) { cq++; cr=0; }
// TODO = occ control
        mcsr_cnt[row_panel+1] = CEIL(cr,BW)+1;

    }

////gettimeofday(&tt1, NULL);
    // prefix-sum
    for(int i=1; i<=npanel;i++)
        mcsr_cnt[i] += mcsr_cnt[i-1];
    //mcsr_e[0] = 0;
    mcsr_e[BH * mcsr_cnt[npanel]] = ne;

////gettimeofday(&tt2, NULL);

#pragma omp parallel for num_threads(68) schedule(dynamic, 1)
    for(int row_panel=0; row_panel<nr/BH; row_panel++) {
        int tid = omp_get_thread_num();
        if(mcsr_chk[row_panel] == 0) {
            int i, j;
            int flag = 0;
            int cq=0, cr=0;
            for(int stride = 1; stride <= BH/2; stride*=2, flag=1-flag);
            int base = (mcsr_cnt[row_panel]*BH);
            int mfactor = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
            int weight=1;

            // mcsr_e making
            for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
                if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
                else {
                    int reminder = (csr_e1[flag][i-1]&31);
                    if(weight >= THRESHOLD) {
                        cr++;
                        bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
                        for(j=i-weight; j<=i-1; j++) {
                            mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
                        }
                    } else {
                        //bv[tid][csr_e1[flag][i-1]>>5] &= (~0 - (1<<reminder));
                        bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
                    }
                    if(cr == BW) { cq++; cr=0;}
                    weight = 1;
                }
            }

//fprintf(stderr, "inter : %d\n", i);

            int reminder = (csr_e1[flag][i-1]&31);
            if(weight >= THRESHOLD) {
                cr++;
                bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
                for(j=i-weight; j<=i-1; j++) {
                    mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
                }
            } else {
                bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
            }
            // reordering
            int delta = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
            int base0 = mcsr_cnt[row_panel]*BH;
            for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
                int base = base0+(i-row_panel*BH)*delta;
                int dpnt = mcsr_e[base] = csr_v[i];
                for(int j=1;j<delta;j++) {
                    mcsr_e[base+j] += mcsr_e[base+j-1];
                }
                int spnt=mcsr_e[mcsr_cnt[row_panel]*BH + (mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel])*(i - row_panel*BH + 1) - 1];

                avg0[tid] += csr_v[i+1] - spnt;
                for(j=csr_v[i]; j<csr_v[i+1]; j++) {
                    int k = csr_e0[j];
                    if((bv[tid][k>>5]&(1<<(k&31)))) {
                        csr_e[dpnt] = csr_e0[j];
                        csr_ev[dpnt++] = csr_ev0[j];
                    } else {
                        csr_e[spnt] = csr_e0[j];
                        csr_ev[spnt++] = csr_ev0[j];
                    }
                }
            }
        } else {
            int base0 = mcsr_cnt[row_panel]*BH;
            memcpy(&mcsr_e[base0], &csr_v[row_panel*BH], sizeof(int)*BH);
            avg0[tid] += csr_v[(row_panel+1)*BH] - csr_v[row_panel*BH];
            int bidx = csr_v[row_panel*BH];
            int bseg = csr_v[(row_panel+1)*BH] - bidx;
            memcpy(&csr_e[bidx], &csr_e0[bidx], sizeof(int)*bseg);
            memcpy(&csr_ev[bidx], &csr_ev0[bidx], sizeof(FTYPE)*bseg);

        }
    }


    for(int i=0;i<NTHREAD;i++)
        avg += avg0[i];
    avg /= (double)nr;

////gettimeofday(&tt3, NULL);

    for(int i=0;i<nr;i++) {
        int idx = (mcsr_cnt[i>>LOG_BH])*BH + (mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH])*((i&(BH-1))+1);
        int diff = csr_v[i+1] - mcsr_e[idx-1];
        double r = ((double)diff - avg);
        vari += r * r;

        if(diff >= STHRESHOLD) {
            int pp = (diff) / STHRESHOLD;
            for(int j=0; j<pp; j++) {
                special[special_p] = i;
                special2[special_p] = j * STHRESHOLD;
                special_p++;
            }
        }
    }
    vari /= (double)nr;

    for(int i=0;i<NTHREAD;i++)
        free(bv[i]);
    for(int i=0;i<2;i++) {
        free(csr_e1[i]);
        free(coo[i]);
    }
    free(bv); free(csr_e1); free(coo);

    struct InspectorMetadata<FTYPE> meta;
    meta.nThread = NTHREAD;
    meta.npanel = npanel;
    meta.nr = nr;

    meta.mcsr_e = mcsr_e;
    meta.mcsr_cnt = mcsr_cnt;
    meta.mcsr_chk = mcsr_chk;

    meta.row_ptrs_padded = csr_v;
    meta.col_indices_reordered = csr_e;
    meta.values_reordered = csr_ev;

    meta.avg = avg;
    meta.vari = vari;

    meta.special = special;
    meta.special2 = special2;
    meta.special_p = special_p;

    return meta;
}

void execute(
    const InspectorMetadata<FTYPE>& meta,
    int nr0, int nc, int sc,
    int* row_ptrs, int* col_indices, FTYPE* values,
    FTYPE* vin,
    FTYPE* vout,
    int ASpT_block_height
) {
    int* csr_e0 = col_indices;
    FTYPE* csr_ev0 = values;

    int NTHREAD = meta.nThread;
    int npanel = meta.npanel;
    int nr = meta.nr;

    int* mcsr_e = meta.mcsr_e;
    int* mcsr_cnt = meta.mcsr_cnt;
    int* mcsr_chk = meta.mcsr_chk;

    int* special = meta.special;
    int* special2 = meta.special2;
    int special_p = meta.special_p;

    int* csr_v = meta.row_ptrs_padded;
    int* csr_e = meta.col_indices_reordered;
    FTYPE* csr_ev = meta.values_reordered;

    double avg = meta.avg;
    double vari = meta.vari;

    csr_v = assume_aligned<64>(csr_v);
    csr_e = assume_aligned<64>(csr_e);
    csr_ev = assume_aligned<64>(csr_ev);
	//__assume_aligned(mcsr_cnt, 64);
	//__assume_aligned(mcsr_e, 64);
	//__assume_aligned(mcsr_list, 64);
    vin = assume_aligned<64>(vin);
    vout = assume_aligned<64>(vout);

     memset(vout, 0, sizeof(FTYPE) * nr0 * sc);

     if (vari < 5000 * 1 / 1 * 1) {

////begin
#pragma ivdep
#pragma vector aligned
#pragma temporal (vin)
#pragma omp parallel for schedule(dynamic, 1)
             for (int row_panel = 0; row_panel < nr / BH; row_panel++) {
                 //dense
                 int stride;
                 for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++) {

                     for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {
                         int dummy = mcsr_cnt[row_panel] * BH +
                                     (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                         int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                         int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                         int j;
                         for (j = loc1; j < interm; j += 8) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vin:_MM_HINT_T1
#pragma temporal (vin)
                             for (int k = 0; k < sc; k++) {
                                 vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                                    + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                                    + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                                    + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                                    + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                                    + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                                    + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                                    + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                             }
                         }
                         for (; j < loc2; j++) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vout:_MM_HINT_T1
#pragma temporal (vout)
                             for (int k = 0; k < sc; k++) {
                                 vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                             }
                         }
                     }

                 }
                 //sparse
                 for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {

                     int dummy = mcsr_cnt[row_panel] * BH +
                                 (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                     int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

//printf("(%d %d %d %d %d)\n", i, csr_v[i], loc1, csr_v[i+1], loc2);
//printf("%d %d %d %d %d %d %d\n", i, dummy, stride, csr_v[i], loc1, csr_v[i+1], loc2);


                     int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                     int j;
                     for (j = loc1; j < interm; j += 8) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vin:_MM_HINT_T1
#pragma temporal (vin)
                         for (int k = 0; k < sc; k++) {
                             vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                                + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                                + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                                + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                                + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                                + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                                + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                                + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                         }
                     }
                     for (; j < loc2; j++) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vout:_MM_HINT_T1
#pragma temporal (vout)
                         for (int k = 0; k < sc; k++) {
                             vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                         }
                     }
                 }
             }
////end
     } else { // big var
////begin
#pragma ivdep
#pragma vector aligned
#pragma temporal (vin)
#pragma omp parallel for schedule(dynamic, 1)
             for (int row_panel = 0; row_panel < nr / BH; row_panel++) {
                 //dense
                 int stride;
                 for (stride = 0; stride < mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1; stride++) {

                     for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {
                         int dummy = mcsr_cnt[row_panel] * BH +
                                     (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                         int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                         int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                         int j;
                         for (j = loc1; j < interm; j += 8) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vin:_MM_HINT_T1
#pragma temporal (vin)
                             for (int k = 0; k < sc; k++) {
                                 vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                                    + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                                    + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                                    + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                                    + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                                    + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                                    + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                                    + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                             }
                         }
                         for (; j < loc2; j++) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vout:_MM_HINT_T1
#pragma temporal (vout)
                             for (int k = 0; k < sc; k++) {
                                 vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                             }
                         }
                     }

                 }
                 //sparse
                 for (int i = row_panel * BH; i < (row_panel + 1) * BH; i++) {

                     int dummy = mcsr_cnt[row_panel] * BH +
                                 (i & (BH - 1)) * (mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel]) + stride;
                     int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy + 1];

                     loc1 += ((loc2 - loc1) / STHRESHOLD) * STHRESHOLD;

                     int interm = loc1 + (((loc2 - loc1) >> 3) << 3);
                     int j;
                     for (j = loc1; j < interm; j += 8) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vin:_MM_HINT_T1
#pragma temporal (vin)
                         for (int k = 0; k < sc; k++) {
                             vout[i * sc + k] = vout[i * sc + k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                                + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                                + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                                + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                                + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                                + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                                + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                                + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                         }
                     }
                     for (; j < loc2; j++) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vout:_MM_HINT_T1
#pragma temporal (vout)
                         for (int k = 0; k < sc; k++) {
                             vout[i * sc + k] += csr_ev[j] * vin[csr_e[j] * sc + k];
                         }
                     }
                 }
             }
#pragma ivdep
#pragma vector aligned
#pragma temporal (vin)
#pragma omp parallel for schedule(dynamic, 1)
             for (int row_panel = 0; row_panel < special_p; row_panel++) {
                 int i = special[row_panel];

                 int dummy = mcsr_cnt[i >> LOG_BH] * BH +
                             ((i & (BH - 1)) + 1) * (mcsr_cnt[(i >> LOG_BH) + 1] - mcsr_cnt[i >> LOG_BH]);

                 int loc1 = mcsr_e[dummy - 1] + special2[row_panel];
                 int loc2 = loc1 + STHRESHOLD;

                 //int interm = loc1 + (((loc2 - loc1)>>3)<<3);
                 int j;

                 FTYPE temp_r[sc];
                 memset(temp_r, sc * sizeof(FTYPE), 0);

                 for (j = loc1; j < loc2; j += 8) {
#pragma ivdep
#pragma vector nontemporal (csr_ev)
#pragma prefetch vin:_MM_HINT_T1
#pragma temporal (vin)
                     for (int k = 0; k < sc; k++) {
                         temp_r[k] = temp_r[k] + csr_ev[j] * vin[csr_e[j] * sc + k]
                                     + csr_ev[j + 1] * vin[csr_e[j + 1] * sc + k]
                                     + csr_ev[j + 2] * vin[csr_e[j + 2] * sc + k]
                                     + csr_ev[j + 3] * vin[csr_e[j + 3] * sc + k]
                                     + csr_ev[j + 4] * vin[csr_e[j + 4] * sc + k]
                                     + csr_ev[j + 5] * vin[csr_e[j + 5] * sc + k]
                                     + csr_ev[j + 6] * vin[csr_e[j + 6] * sc + k]
                                     + csr_ev[j + 7] * vin[csr_e[j + 7] * sc + k];
                     }
                 }
#pragma ivdep
                 for (int k = 0; k < sc; k++) {
#pragma omp atomic
                     vout[i * sc + k] += temp_r[k];
                 }
             }

         }
}

void free(struct InspectorMetadata<FTYPE> &meta) {
    free(meta.mcsr_e);
    free(meta.mcsr_cnt);
    free(meta.mcsr_chk);
    free(meta.row_ptrs_padded);
    free(meta.col_indices_reordered);
    free(meta.values_reordered);
    free(meta.special);
    free(meta.special2);
}
