//
// Created by lwilkinson on 5/9/22.
//

#include "sparse_io.h"
#include "template_utils.h"

//
// Created by Kazem on 10/10/19.
//
#include <def.h>
#include <tuple>
#include <mmio.h>
#include <iostream>
#include "sparse_io.h"

namespace sym_lib {

template <typename Scalar>
CSC<Scalar>* read_mtx(std::string fname) {
  FILE* mf = fopen(fname.c_str(), "r");
  if (!mf)
    exit(1);

  MM_typecode mcode;
  if (mm_read_banner(mf, &mcode) != 0) {
    std::cerr << "Error processing matrix banner\n";
    fclose(mf);
    return nullptr;
  }

  int m, n, nnz;
  if (mm_read_mtx_crd_size(mf, &m, &n, &nnz) != 0)
    exit(1);
  CSC<Scalar>* A = new CSC<Scalar>(m, n, nnz);
  int* J = new int[nnz]();

  A->stype = mm_is_symmetric(mcode) ? -1 : 0;

  // Copy matrix data into COO format
  for (int i = 0; i < nnz; i++) {
    int status;

    if constexpr(std::is_same_v<Scalar, float>)
        status = fscanf(mf, "%d %d %f\n", (A->i) + i, &J[i], (A->x) + i);
    else
        status = fscanf(mf, "%d %d %lg\n", (A->i) + i, &J[i], (A->x) + i);

    if (status == EOF) {
      std::cerr << "Failed to load matrix at " << i + 1 << "th element\n";
      fclose(mf);
      delete (A);
      return nullptr;
    }
    (A->i)[i]--;
    J[i]--;
  }
  int i = 0, j;
  int index = 0, cur;
  for (; i < nnz; i++) {
    A->p[index] = i;
    cur = J[i];
    for (j = i + 1; j < nnz; j++) {
      if (J[j] != cur)
        break;
      else
        i++;
    }
    index += 1;
  }
  A->p[n] = nnz;
  delete[] J;

  A->m = m;
  A->n = n;
  A->nnz = nnz;
  return A;
}



template <typename Scalar>
void CSC_to_mtx(std::string fname, CSC<Scalar>* A) {
  FILE* fp = fopen(fname.c_str(), "w");

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(fp, matcode);
  mm_write_mtx_crd_size(fp, A->m, A->n, A->nnz);

  for (int i = 0; i < A->n; i++)
    for (int j = A->p[i]; j < A->p[i + 1]; j++)
      fprintf(fp, "%d %d %10.3g\n", A->i[j] + 1, i + 1, A->x[j]);
  fclose(fp);
}

template <typename Scalar>
void BCSC_to_mtx(std::string fname, BCSC<Scalar>* A) {
  FILE* f = fopen(fname.c_str(), "w");

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(f, matcode);
  mm_write_mtx_crd_size(f, A->m, A->n, A->nnz);

  for (int i = 0; i < A->nodes; i++) {
    int index = A->p[i];
    int width = A->supernodes[i + 1] - A->supernodes[i];
    int nrows = (A->p[i + 1] - A->p[i]) / width;

    for (int j = 0; j < width; j++) {
      for (int k = 0; k < nrows; k++) {
        int pos = index + j * nrows + k;
        fprintf(
            f,
            "%d %d %10.3g\n",
            A->i[pos] + 1,
            A->supernodes[i] + j + 1,
            A->x[pos]);
      }
    }
  }
  fclose(f);
}

template <typename Scalar>
CSC<Scalar>* convert_to_one_based(const CSC<Scalar>* A) {
  CSC<Scalar>* A_one = new CSC<Scalar>(A->m + 1, A->n + 1, A->nnz, A->is_pattern);
  if (A->is_pattern)
    for (int j = 0; j < A->nnz; ++j) {
      A_one->i[j] = A->i[j] + 1;
    }
  else
    for (int j = 0; j < A->nnz; ++j) {
      A_one->i[j] = A->i[j] + 1;
      A_one->x[j] = A_one->x[j];
    }

  for (int k = 0; k < A->n + 1; ++k) {
    A_one->p[k + 1] = A->p[k];
  }
  A_one->m--;
  A_one->n--;
  return A_one;
}

template <typename Scalar>
CSC<Scalar>* dense_to_csc(int rows, int cols, double** val) {
  int n_nz = 0;
  int* nz_col = new int[cols]();
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (val[i][j] != 0) {
        n_nz++;
        nz_col[j]++;
      }
    }
  }
  CSC<Scalar>* out = new CSC<Scalar>(rows, cols, n_nz);
  for (int k = 0; k < cols; ++k) {
    out->p[k + 1] = out->p[k] + nz_col[k];
  }
  assert(out->p[cols] == n_nz);
  n_nz = 0;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (val[i][j] != 0) {
        out->i[n_nz] = i;
        out->x[n_nz] = val[i][j];
        n_nz++;
      }
    }
  }
  delete[] nz_col;
  return out;
}

template <typename Scalar>
void _print_csc(
    int fd,
    std::string beg,
    size_t n,
    int* Ap,
    int* Ai,
    Scalar* Ax) {
  dprintf(fd, "%s\n", beg.c_str());
  int nnz = n > 0 ? Ap[n] : 0;
  dprintf(fd, "%zu %zu %d\n", n, n, nnz);
  for (int i = 0; i < n; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      double x = Ax != NULLPNTR ? Ax[j] : 0;
      dprintf(fd, "%d %d %.12f", Ai[j] + 1, i + 1, x);
      //    std::cout<<Ai[j]+1<<" "<<i+1<<" "<<std::setprecision(12)<< x;
      if (j + 1 != Ap[n])
        dprintf(fd, "\n");
    }
  }
}

template <typename Scalar>
void print_csc(int fd, std::string beg, CSC<Scalar>* A) {
  _print_csc<Scalar>(fd, beg, A->n, A->p, A->i, A->x);
}

template <typename Scalar>
void print_dense(int fd, Dense<Scalar>* A) {
  for (int i = 0; i < A->row; i++) {
    for (int j = 0; j < A->col; j++) {
      dprintf(fd, "%f ", A->a[i + j * A->row]);
    }
    dprintf(fd, "\n");
  }
}

void print_level_set(std::string beg, int n, int* level_ptr, int* level_set) {
  std::cout << beg;
  for (int i = 0; i < n; ++i) {
    for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j) {
      std::cout << level_set[j] << ",";
    }
    std::cout << "\n";
  }
}

void print_hlevel_set(
    std::string beg,
    int n,
    const int* level_ptr,
    const int* level_part_ptr,
    int* level_set) {
  std::cout << beg;
  for (int i = 0; i < n; ++i) {
    for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j) {
      for (int k = level_part_ptr[j]; k < level_part_ptr[j + 1]; ++k) {
        std::cout << level_set[k] << ",";
      }
      std::cout << "; \n";
    }
    std::cout << "\n\n";
  }
}

INSTANTIATE_FOR_SUPPORTED_SCALARS(read_mtx);
INSTANTIATE_FOR_SUPPORTED_SCALARS(CSC_to_mtx);
INSTANTIATE_FOR_SUPPORTED_SCALARS(BCSC_to_mtx);
INSTANTIATE_FOR_SUPPORTED_SCALARS(dense_to_csc);
//INSTANTIATE_FOR_SUPPORTED_SCALARS(print_csc);
//INSTANTIATE_FOR_SUPPORTED_SCALARS(print_dense);

}
