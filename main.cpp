#include "mpi.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#define MAX_P 10
#define EPS 1e-15
//#define SLEEP 60
void
inline mult_block (double *a, double *b, double *c, int h, int w, int l);
int inline solve (int n, int m, const char *name, int p, int k, MPI_Comm G);
int inline rows_p_process (int Nr, int p, int k);
int inline rows_p_process (int n, int m, int p, int k);
void inline matrix_mult_matrix (double *a, double *b, double *d/*матрица в кото-ой хранится результат*/, int n, int m, int p, int k, MPI_Comm G);
void inline local_to_global (int i, int j, int /*n*/, int m, int p, int k, int &gi, int &gj);
int inline init_matrix (double *a, int n, int m, int p, int k, double (*f)(int, int));
int inline read_matrix (double *a, double *buf, int n, int m, const char *name, int p, int k);
int inline print_matrix (double *a, double *c, int n, int m, int p, int k, MPI_Comm G);
int inline solve (int n, int m, const char *name, int p, int k, MPI_Comm G);
int inline find_sender (int Nr /*кол-во блочных строк в матрице*/, int p);
int inline find_sender (int n, int m, int p);
double inline norm (double *a, int n, int m, int p, int k);
int inline get (double *a, double *E, int i_b, int j_b, int a_height, int n, int m, int &bl_h, int &bl_w);
int inline set (double *E, double *a, int i_b, int j_b, int a_height, int n, int m);
int Jordan (double *a, double *b, double *c, int n, int m, int p, int k, MPI_Comm G);
double inline block_norm (double *E, int h/*высота блока*/, int w/*длина  блока*/);
int inline inverse_block (double *a, double *b, double norm, int n);
void inline to_i (double *A, int n);
int inline num_block_rows (int n, int m);
int inline get_from_row (double *c, double *E, int j_b, int shift /*t-ый блок при обмене строки лежит первым*/, int c_height, int n, int m, int &bl_h, int &bl_w); //взять из буффера нужный блок
void inline fill_row (double *c /*буфер*/, double *a, double *b, int n, int m, int t, int row /*номер ряда откуда берем строки*/, int p, int k); //заполнить строку а и b из буффера с
void inline gather_row (double *a, double *b, double *c /*буфер*/, int n, int m, int t, int row /*номер ряда откуда берем строки*/, int p, int k);
void inline mult_substr_block(double *a, double *b, double *c, int h, int w, int o); //из c -= a * b
void minus_i (double *d, int n, int m, int p, int k);
struct PivotMin
{
  double min; //минимальная норма обратного элемента в данном процессе
  int k; //номер процесса
  int min_row; //локальный номер строки с главным элементом
  int non_sing; //cуществует невырожденный начальный элемент в среди строк процесса
  
};

void pivot_func (void *a, void *b, int *len, MPI_Datatype */*type*/);
void pivot_op (PivotMin *a, PivotMin *b);

double f (int i, int j)
{
#ifdef HILBERT
  return 1./(i + j + 1);
#endif
  return fabs (i - j);
  /*int n = 4992;
  if (j == n - 1 - i)
   return i + 1;
  else return 0;*/
}

double f_i (int i, int j)
{
  if (i == j)
    return 1;
  return 0;
}
int main (int argc, char *argv[])
{
  int p, k;
  MPI_Comm G = MPI_COMM_WORLD;
  MPI_Init (&argc, &argv);
#ifdef SLEEP
  sleep (SLEEP);
#endif
  MPI_Comm_size (G, &p);
  MPI_Comm_rank (G, &k);
  int n, m;
  char *name = 0;
  if (argc > 4 || argc < 3 || (n = atoi (argv[1])) == 0 || (m = atoi (argv[2])) == 0)
    {
      if (k == 0)
        printf ("usage:%s n m [<file>]\n", argv[0]);
      MPI_Finalize ();
      return 1;
    }
  
  if (argc >= 4) name = argv[3];
  int ret = 0;
  if (solve (n, m, name, p, k, MPI_COMM_WORLD))
    {
      ret = 2;
    }
  MPI_Finalize ();
  return ret;
}

int rows_p_process (int n, int m, int p, int k)
{
  int Nr = n % m? n / m + 1: n / m; //кол-во блочных строк
  return rows_p_process (Nr, p, k);
}

int rows_p_process (int Nr, int p, int k)
{
  int sender = find_sender (Nr, p);

  int RpP; //кол-во блочных строк на процесс
  if (!(Nr % p))
    RpP =  Nr / p;
  else
    {
      if (k <= sender)
        RpP = Nr / p + 1;
      else
        RpP = Nr / p;
    }
  return RpP;
}

void inline
local_to_global (int i, int j, int /*n*/, int m, int p, int k, int &gi, int &gj)
{
  gi = ((i / m) * p + k) * m + i % m;
  gj = j;
}
int num_block_rows (int n, int m)
{
  return n % m? n / m + 1: n / m;
}
int init_matrix (double *a, int n, int m, int p, int k, double (*f) (int, int))
{
  int Nr = num_block_rows (n, m);
  int RpP = rows_p_process (n, m, p, k);
  int l = m;
  if (!(Nr - 1 - k) % p) //последняя полоса принадлежит этому потоку
    {
      l = n - m * (Nr - 1);
    }
  int height = m * (RpP - 1) + l;//высота хранящейся полосы
  double *p_a = a;
  for (int i = 0; i < height; i++, p_a += n)
    {
      for (int j = 0; j < n; j++)
        {
          int glob_i, glob_j;
          local_to_global (i, j, n, m, p, k, glob_i, glob_j);
          p_a[j] = f (glob_i, glob_j);
        }
    }
  return 0;
}

void 
inline mult_substr_block (double *a, double *b, double *c, int h, int w, int l)
{
  double *pa, *pb, *pc;
  double s00, s01, s02, s10, s11, s12, s20, s21, s22;
  int m, i, j;

  if (h % 3 == 0 && w % 3 == 0 && l % 3 == 0)
    {
      for (m = 0, pc = c; m < l; m += 3, pc += 3)
        {
          for (i = 0, pb = b + m; i < h; i += 3)
            {
              pa = a + i * w;
              s00 = s01 = s02 = s10 = s11 = s12 = s20 = s21 = s22 = 0.;
              for (j = 0; j < w; j++, pa++)
                {
                  s00 += pa[0] * pb[j * l];
                  s01 += pa[0] * pb[j * l + 1];
                  s02 += pa[0] * pb[j * l + 2];
                  s10 += pa[w] * pb[j * l];
                  s11 += pa[w] * pb[j * l + 1];
                  s12 += pa[w] * pb[j * l + 2];
                  s20 += pa[2 * w] * pb[j * l];
                  s21 += pa[2 * w] * pb[j * l + 1];
                  s22 += pa[2 * w] * pb[j * l + 2];
                }
              pc[i * l] -= s00;
              pc[i * l + 1] -= s01;
              pc[i * l + 2] -= s02;
              pc[(i + 1) * l] -= s10;
              pc[(i + 1) * l + 1] -= s11;
              pc[(i + 1) * l + 2] -= s12;
              pc[(i + 2) * l] -= s20;
              pc[(i + 2) * l + 1] -= s21;
              pc[(i + 2) * l + 2] -= s22;
            }
          }
        }   
      else
        {
          for (m = 0, pc = c; m < l; m++, pc++)
            {
              for (i = 0, pb = b + m; i < h; i++)
                {
                  double sum = 0;
                  pa = a + i * w;
                  for (j = 0; j < w; j++, pa++)
                    {
                      sum += pa[0] * pb[j * l];
                    }
                  pc[i * l] -= sum;
                }
            }
        }
}


int read_matrix (double *a, double *buf, int n, int m, const char *name, int p, int k)
{
  int loc_error = 0; // здесь хранится ошибка считывания
  MPI_Comm G = MPI_COMM_WORLD;
  int Nr = num_block_rows(n, m); //кол-во блочных строк
  int sender = find_sender (Nr, p); //номер процесса, который всем будет отсылать
  
  int NrR = 0; //кол-во строк полученных в результате обмена
  int RpP; //кол-во блочных строк на процесс
  if (!(Nr % p))
    RpP =  Nr / p;
  else
    {
      if (k <= sender)
        RpP = Nr / p + 1;
      else
        RpP = Nr / p;
    }
    
  int len = RpP * m * n; //размер блока матрицы, который принадлежит процессу
  FILE *fp = 0;
  
  if (k == sender)
    {
      if (!(fp = fopen(name, "r")))
        loc_error = 1;
    }
  
  MPI_Bcast (&loc_error, 1, MPI_INT, sender, G);
  if (loc_error) return -1;
  memset (a, 0, len * sizeof (double));
  memset (buf, 0, 2 * m * n * sizeof (double));

  for (int i = 0; i < Nr; i++)
    {
      int reciever = i % p; //кому принадлежит строка
      int act_height = m > n - m * i? n - m * i: m;
      int send_len = act_height * n; //длина посылаемого буфера
      if (k == sender)
        {
          for (int j = 0; j < send_len; j++)
            {
              if (fscanf (fp, "%lf", buf + j) != 1)
                {
                  loc_error = 2;
                  break;
                }
            }
          if (reciever != sender)
            MPI_Send (buf, send_len, MPI_DOUBLE, reciever, 0, G);
          else
            {
              double *p_a = a + m * n * NrR;
              NrR++;
              memcpy (p_a, buf, send_len * sizeof (double));
            }
        }
      else
        {
          if (k == reciever)
            {
              MPI_Status s;
              MPI_Recv (buf, send_len, MPI_DOUBLE, sender, 0, G, &s);
              memcpy (a + NrR * m * n, buf, send_len * sizeof (double));
              NrR++;
            }
        }
    }
  MPI_Bcast (&loc_error, 1, MPI_INT, sender, G);
  if (k == sender)
    fclose (fp);
  if (loc_error) return -2;
  return 0;
}

int print_row (int n, int height, double *buf, int nm)
{
  for (int i = 0; i < height; i++, buf += n)
    {
      for (int j = 0; j < nm; j++)
        {
          printf ("%.2f\t", buf[j]);
        }
      printf ("\n");
    }
  return 0;
}

int print_matrix (double *a, double *c, int n, int m, int p, int k, MPI_Comm G)
{
  int owner;
  int NrP; // кол-во блочных строк которые нужно вывести
  int nm = n > MAX_P? MAX_P: n;
  NrP = nm % m? nm / m + 1: nm / m;
  int NrR = 0; // кол-во переданных процессом строк
  for (int i = 0; i < NrP; i++)
    {
      int act_height = nm - i * m > m? m: nm - i * m; 
      int send_len = act_height * nm;
      owner = i % p;
      if (k == 0)
        {
          if (owner == k)
            {
              print_row (n, act_height, a + m * NrR * n, nm);
              NrR++;
            }
          else
            {
              MPI_Status st;
              MPI_Recv (c, send_len, MPI_DOUBLE, owner, 0, G, &st);
              print_row (nm, act_height, c, nm);
            }
        }
      else
        {
          if (k == owner)
            {
             
              double *p_c = c;
              double *p_a = a + m * NrR * n;             
              for (int i = 0; i < act_height; i++, p_a += n)
                {
                  for (int j = 0; j < nm; j++, p_c++)
                    *p_c = p_a[j];
                }
              MPI_Send (c, send_len, MPI_DOUBLE, 0, 0, G);
              NrR++;
            }
        }
    }
  return 0;
}

int solve (int n, int m, const char *name, int p, int k, MPI_Comm G)
{
  double *a, *c, *b, *d;
  
  /*int Nr = n % m? n / m + 1: n / m; //кол-во блочных строк
  int sender = Nr % p - 1; //номер процесса, который всем будет отсылать
  if (sender < 0)
    sender += p;*/
    
  int loc_err = 0;
  int glob_err = 0;
    
  int RpP = rows_p_process (n, m, p, k); //кол-во блочных строк на процесс
  /*if (!(Nr % p))
    RpP =  Nr / p;
  else
    {
      if (k <= sender)
        RpP = Nr / p + 1;
      else
        RpP = Nr / p;
    }*/
    
  int len = RpP * m * n; //размер блока матрицы, который принадлежит процессу
  if (!(a = new double [len])) loc_err = 1; //выделяем с учетом буфера
  if (!(b = new double [len + 2 * m * n])) loc_err = 1;
  if (!(d = new double [len])) loc_err = 1;
  c = b + len;
  MPI_Allreduce (&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
  if (glob_err)
    {
      if (k == 0)
        printf ("Not enough memory!\n");
      
      if (a) delete []a;
      if (b) delete []b;
      if (d) delete []d;
      return -1;
    }
  double loc_t, glob_t;
  if (name)
    {
      glob_err = read_matrix (a, c, n, m, name, p, k);
      if (glob_err)
        {
          if (k == 0)
            {
              switch (glob_err)
                {
                  case -1: printf ("cannot open %s\n", name);
                           break;
                  case -2: printf ("cannot read %s\n", name);
                           break;
                  default: printf ("unknown error %d in %s\n", glob_err, name);
                }
            }
          delete[] a;
          delete[] b;
          delete[] d;
          return -2;
        }
    }
  else
    {
      init_matrix (a, n, m, p, k, f);
    }
  //matrix_mult_matrix (a, b, d, n, m, p, k, G);
  if (k == 0)
    {
      printf ("A\n");
    }
  print_matrix (a, c, n, m, p, k, G);
  init_matrix (b, n, m, p, k, f_i);
  /*if (k == 0)
    {
      printf ("B\n");
    }
  print_matrix (b, c, n, m, p, k, G);*/
  
  double loc_norm;
  double glob_norm;
  
  
  //memset (b, 0, len * sizeof (double));
  loc_t = MPI_Wtime ();
  if ((glob_err = Jordan (a, b, c, n, m, p, k, G)))
   {
     if (k == 0)
       {
         switch (glob_err)
           {
              case -3:
                printf ("not enough memory for block\n");
                break;
              case -2:
                printf ("singular matrix\n");
                break;
              default:
                printf ("unknown error: %d\n", glob_err);
            }
        }
      delete[] a;
      delete[] b;
      delete[] d;
      return -1;
              
   }
  loc_t = MPI_Wtime () - loc_t;
  //printf ("process: %d, loc_time: %.2f\n", k, loc_t);
  /*if (k == 0)
    printf ("after Jordan a\n");
  print_matrix (a, c, n, m, p, k, G);*/
  MPI_Allreduce (&loc_t, &glob_t, 1, MPI_DOUBLE, MPI_MAX, G);
  if (k == 0)
    {
      printf ("glob_time: %.2f\n", glob_t);
      printf ("inverse matrix:\n\n");
    }
  print_matrix (b, c, n, m, p, k, G);
  //восстанавливаем матрицу a
  if (name)
    {
      glob_err = read_matrix (a, c, n, m, name, p, k);
      if (glob_err)
        {
          if (k == 0)
            {
              switch (glob_err)
                {
                  case -1: printf ("cannot open for residual %s\n", name);
                           break;
                  case -2: printf ("cannot read for residual %s\n", name);
                           break;
                  default: printf ("unknown error in residual %d in %s\n", glob_err, name);
                }
            }
          delete[] a;
          delete[] b;
          delete[] d;
          return -2;
        }
    }
  else
    {
      init_matrix (a, n, m, p, k, f);
    }
  //memset (d, 0, len * sizeof (double));
#ifdef HILBERT
  matrix_mult_matrix (a, b, d, n, m, p, k, G);
  minus_i (d, n, m, p, k);
 //print_matrix (d, c, n, m, p, k, G);
  loc_norm = norm (d, n, m, p, k);
  MPI_Allreduce (&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, G);
  if (k == 0)
    printf ("residual: %e\n", glob_norm);
#else
  if (p != 1)
    {
      matrix_mult_matrix (a, b, d, n, m, p, k, G);
      minus_i (d, n, m, p, k);
     //print_matrix (d, c, n, m, p, k, G);
      loc_norm = norm (d, n, m, p, k);
      MPI_Allreduce (&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, G);
      if (k == 0)
        printf ("residual: %e\n", glob_norm);
    }
  else
    {
      if (k == 0)
        printf ("p == 1!\n");
    }
#endif
  delete[] a;
  delete[] b;
  delete[] d;
  return 0;
}

int find_sender (int n, int m, int p)
{
  int Nr = num_block_rows (n, m);
  return find_sender (Nr, p);
}
int find_sender (int Nr /*кол-во блочных строк в матрице*/, int p)
{
  int sender = Nr % p - 1;
  if (sender < 0)
    sender += p;
  return sender;
}

void matrix_mult_matrix (double *a, double *b, double *d/*матрица в кото-ой хранится результат*/, int n, int m, int p, int k, MPI_Comm G)
{
  int Nr = num_block_rows (n, m);
  int l_h = n - (Nr - 1) * m; // высота последний строки процесса sender
  
  int sender = find_sender (Nr, p);
  int NrP = rows_p_process (n, m, p, k); //кол-во блочных строк принадлежащих процессу
  int i, j, iter, l, i1, i2;
  int len = (Nr % p? Nr / p + 1: Nr / p) * m * n; //объём памяти отведенной процессу k  (считаем максимальный для пересылок)
  int j1, j2;
  double *pa, *pb, *pd;
  
  double s00, s01, s02, s10, s11, s12, s20, s21, s22;
  s00 = s01 = s02 = s10 = s11 = s12 = s20 = s21 = s22 = 0.;
  i1 = 0;
  i2 = (NrP - 1) * m; //указатель на начало последней блочной строки принадлежащей процессу k
  
  int u, v, z;
  
  //printf ("I'm %d! i1:%d i2:%d\n", k, i1, i2);
  //кол-во обычных строк принадлежащих процессу k
  int a_height = (NrP - 1) * m;
  if (k == sender)
    a_height += l_h;
  else
    a_height += m;
  
  memset (d, 0, a_height * n * sizeof (double)); 
  
  
  int src = (k + 1) % p; //процесс, которому мы передаем строку при сдвиге
  int dst = (k - 1 + p) % p; //процесс, от которого получаем строку при сдвиге
  
  for (iter = 0; iter < p; iter++)
    {
      int whose_rows = (k + iter) % p; //процесс, чьи строки лежат в правой части
      j1 = 0; //указатель на первую блочную строку процесса whose_rows
      int NrPw = rows_p_process (n, m, p, whose_rows); //кол-во блочных строк процесса whose_rows
      
      j2 = (NrPw - 1) * m; //указатель на последнюю блочную строку матрицы b
      int b_height = j2;//высота участка матрицы, хранимого в b
      b_height += (whose_rows == sender)? l_h: m; //принадлежит ли последняя строка потоку whose_rows
      for (j = 0, pb = b; j < n; j += m)
        {
           int jn = j + m > n? n: j + m; //край правый край текущего блочного столбца
           for (i = i1; i <= i2; i+=m)
             { 
               int in = i + m > a_height? a_height: i + m; //нижний край текущей блочной строки
               int bl_ind_a; //index of a block to multiply
               for (l = j1, bl_ind_a = whose_rows; l <= j2; l += m, bl_ind_a += p)
                 {
                   int ln = l + m > b_height? b_height: l + m;
                   // (i, whose_rows + l) x (l, j)
                   if ((ln - l) % 3 == 0 && (in - i) % 3 == 0 && (jn - j) % 3 == 0 )
                     {  
                       for (v = j, pd = d + j; v < jn; v += 3, pd += 3)
                        {
                          for (u = i, pb = b + v; u < in; u += 3)
                            {
                              pa = a + bl_ind_a * m + u * n;
                              s00 = s01 = s02 = s10 = s11 = s12 = s20 = s21 = s22 = 0.;
                              for (z = l; z < ln; z++, pa++)
                                {
                                  s00 += pa[0] * pb[z * n];
                                  s01 += pa[0] * pb[z * n + 1];
                                  s02 += pa[0] * pb[z * n + 2];
                                  s10 += pa[n] * pb[z * n];
                                  s11 += pa[n] * pb[z * n + 1];
                                  s12 += pa[n] * pb[z * n + 2];
                                  s20 += pa[2 * n] * pb[z * n];
                                  s21 += pa[2 * n] * pb[z * n + 1];
                                  s22 += pa[2 * n] * pb[z * n + 2];
                                }
                              pd[u * n] += s00;
                              pd[u * n + 1] += s01;
                              pd[u * n + 2] += s02;
                              pd[(u + 1) * n] += s10;
                              pd[(u + 1) * n + 1] += s11;
                              pd[(u + 1) * n + 2] += s12;
                              pd[(u + 2) * n] += s20;
                              pd[(u + 2) * n + 1] += s21;
                              pd[(u + 2) * n + 2] += s22;
                            }
                          }
                       }
                     else
                       {
                         for (v = j, pd = d + j; v < jn; v++, pd ++)
                           {
                              for (u = i, pb = b + v; u < in; u++)
                                {
                                  double s = 0;
                                  pa = a + bl_ind_a * m + u * n;
                                  for (z = l; z < ln; z++, pa++)
                                    {
                                      s += pa[0] * pb[z*n];
                                    }
                                  pd[u * n] += s;
                                }
                           }
					             }
                   }
              }
         }
       MPI_Status status;
       MPI_Sendrecv_replace (b, len, MPI_DOUBLE, dst, 0, src, 0, G, &status);
     }
}

double norm (double *a, int n, int m, int p, int k)
{
  int Nr = num_block_rows (n, m);
  int l_h = n - m * (Nr - 1); //высота последней блочной строки матрицы A
  int sender = find_sender (Nr, p);
  int RpP = rows_p_process (Nr, p, k);
  int a_height =  (RpP - 1) * m;
  if (k == sender)
    a_height += l_h;
  else
    a_height += m;
  double S = 0;
  double max = 0;
  double *p_a = a;
  for (int i = 0; i < a_height; i++, p_a += n)
    {
      S = 0;
      for (int j = 0; j < n; j++)
        S += fabs (p_a [j]);
      
      if (S > max)
       max = S;
    }
  return max;
}

double block_norm (double *E, int h/*высота блока*/, int w/*длина  блока*/)
{
  double S = 0.;
  double max = 0.;
  double *pe = E;
  for (int i = 0; i < h; i++, pe += w)
    {
      S = 0;
      for (int j = 0; j < w; j++)
        S += fabs (pe[j]);
      if (S > max)
        max = S;
    }
  return max;
}

int get_from_row (double *c, double *E, int j_b, int shift, int c_height, int n, int m, int &bl_h, int &bl_w)
{
  get (c, E, 0, j_b - shift, c_height, n, m, bl_h, bl_w);
  return 0;
}
int get (double *a, double *E, int i_b, int j_b, int a_height, int n, int m, int &bl_h, int &bl_w)
{
  int il = i_b * m;//неблочные локальные координаты i
  int jl = j_b * m;//неблочные локальные координаты j
  
  if (il >= a_height || jl >= n)
    return 1;
  bl_h = a_height - il > m? m: a_height - il;  //высота блока E
  bl_w = n - jl > m? m: n - jl; //длина блока Е 
  double *pe = E;
  double *pa = a + il * n;
  int j, i;
  for (i = 0; i < bl_h; i++, pa += n, pe += bl_w)
    {
      for (j = 0; j < bl_w; j++)
        pe[j] = pa [j + jl];
    }
  return 0;
}

int set (double *E, double *a, int i_b, int j_b, int a_height, int n, int m)
{
  int il = i_b * m;//неблочные локальные координаты i
  int jl = j_b * m;//неблочные локальные координаты j
  
  if (il >= a_height || jl >= n)
    return 1;
  int bl_h = a_height - il > m? m: a_height - il;  //высота блока E
  int bl_w = n - jl > m? m: n - jl; //длина блока Е 
  double *pe = E;
  double *pa = a + il * n;
  int j, i;
  for (i = 0; i < bl_h; i++, pa += n, pe += bl_w)
    {
      for (j = 0; j < bl_w; j++)
        pa [j + jl] = pe[j];
    }
  return 0;
}
void pivot_op (PivotMin *a, PivotMin *b)
{
  if (!(a->non_sing)) return;
  if (!(b->non_sing) || a->min < b->min)
    *b = *a;
  return;
}
void pivot_func (void *a, void *b, int *len, MPI_Datatype */*type*/)
{
  PivotMin *pa = (PivotMin *)a;
  PivotMin *pb = (PivotMin *)b;
  for (int i = 0; i < *len; i++)
    {
      pivot_op (pa + i, pb + i);
    }
}

int inline inverse_block (double *a, double *b, double norm, int n)
{
  to_i (b, n);
  
  for (int k = 0; k < n; k++)
    {
      int max_raw = -1;
      double max_norm = -1.;
      
      //нахождение строки с наибольшим модулем начального элемента
      for (int i = k; i < n; i++)
        {
          if (fabs (a[i * n + k]) > max_norm)
            {
              max_norm = fabs (a[i * n + k]);
              max_raw = i;
            }
        }
      //смена строк
      if (max_raw != k)
        {
          double tmp;
          for (int j = k; j < n; j++)
            {
              tmp = a[k*n + j];
              a[k*n + j] = a[max_raw * n + j];
              a[max_raw * n + j] = tmp;
            }
          for (int j = 0; j < n; j++)
            {
              tmp = b[k * n + j];
              b[k * n + j] = b[max_raw * n + j];

              b[max_raw * n + j] = tmp;
            }
        }
      if (fabs (a[k*n + k]) < (EPS * norm) || fabs (norm) < EPS)
        return 1;
           
      //умножение каждого элемента строки A на обратный
      for (int j = k + 1; j < n; j++)
        a[k*n + j] /= a [k * n + k];
          
      //умножение каждого элемента строки B на обратный
      for (int j = 0; j < n; j++)
        b[k*n + j] /= a [k * n + k];
        
      //преобразование матрицы A;
      for (int i = 0; i < k; i++)
        {
          for (int j = k + 1; j < n; j++)
            a[i * n + j] -= a[i * n + k] * a[k * n + j];
        }
             
      for (int i = k + 1; i < n; i++)
        {
          for (int j = k + 1; j < n; j++)
            a[i * n + j] -= a[i * n + k] * a[k * n + j];
        }
            
      //преобразование матрицы B
      for (int i = 0; i < k; i++)
        {
          for (int j = 0; j < n; j++)
            b[i * n + j] -= a[i * n + k] * b[k * n + j];
        }
             
      for (int i = k + 1; i < n; i++)
        {
          for (int j = 0; j < n; j++)
            b[i * n + j] -= a[i * n + k] * b[k * n + j];
        }
    }
    return 0;
}

void
to_i (double *a, int n)
{
  memset (a, 0, n * n * sizeof (double));
  for(int i = 0; i<n; i++)
    a [i * n + i] = 1;
}

void gather_row (double *a, double *b, double *c /*буфер*/, int n, int m, int t, int row /*номер ряда откуда берем строки*/, int p, int k)
{
  int Nr = num_block_rows (n, m);
  int l_h = n - (Nr - 1) * m;
  
  double *pc, *pa, *pb;
  pc = c;
  pa = a + t * m + row * m * n;
  int w_min_row_a = n - t * m;
  int h_min_row = (k + p * row) == (Nr - 1)? l_h: m; 
  /*for (int i = 0; i < h_min_row; i++, pc += w_min_row_a, pa += n)
    {
      for (int j = 0; j < w_min_row_a; j++)
        pc[j] = pa[j];
    }*/
  for (int j = 0; j < w_min_row_a; j++, pc++, pa++)
    {
      for (int i = 0; i < h_min_row; i++)
        pc[i * w_min_row_a] = pa[i * n];
    }
  pc = c + h_min_row * w_min_row_a;
  pb = b + row * m * n;
  /*for (int i = 0; i < h_min_row; i++, pb += n, pc += n)
    {
      for (int j = 0; j < n; j++)
        pc[j] = pb[j];
    }*/
  for (int j = 0; j < n; j++, pc++, pb++)
    {
      for (int i = 0; i < h_min_row; i++)
        pc[i * n] = pb[i * n];
    }
}

void fill_row (double *c /*буфер*/, double *a, double *b, int n, int m, int t, int row /*номер ряда откуда берем строки*/, int p, int k)
{
  int Nr = num_block_rows (n, m);
  int l_h = n - (Nr - 1) * m;
  
  double *pc, *pa, *pb;
  pc = c;
  pa = a + t * m + row * m * n;
  int w_min_row_a = n - t * m;
  int h_min_row = (k + p * row) == (Nr - 1)? l_h: m; 
  for (int j = 0; j < w_min_row_a; j++, pc++, pa++)
    {
      for (int i = 0; i < h_min_row; i++)
        pa[i * n] = pc[i * w_min_row_a];
    }
  pc = c + h_min_row * w_min_row_a;
  pb = b + row * m * n;
  for (int j = 0; j < n; j++, pc++, pb++)
    {
      for (int i = 0; i < h_min_row; i++)
        pb[i * n] = pc[i * n];
    }
}

void mult_block (double *a, double *b, double *c, int h, int w, int l)
{
  double *pa, *pb, *pc;
  double s00, s01, s02, s10, s11, s12, s20, s21, s22;
  int m, i, j;
  
  
  for (m = 0, pc = c; m < l; m++, pc++)
    {
      for (i = 0; i < h; i++)
        {
          pc[i * l] = 0.;
        }
    }
  if (h % 3 == 0 && w % 3 == 0 && l % 3 == 0)
    {
      for (m = 0, pc = c; m < l; m += 3, pc += 3)
        {
          for (i = 0, pb = b + m; i < h; i += 3)
            {
              pa = a + i * w;
              s00 = s01 = s02 = s10 = s11 = s12 = s20 = s21 = s22 = 0.;
              for (j = 0; j < w; j++, pa++)
                {
                  s00 += pa[0] * pb[j * l];
                  s01 += pa[0] * pb[j * l + 1];
                  s02 += pa[0] * pb[j * l + 2];
                  s10 += pa[w] * pb[j * l];
                  s11 += pa[w] * pb[j * l + 1];
                  s12 += pa[w] * pb[j * l + 2];
                  s20 += pa[2 * w] * pb[j * l];
                  s21 += pa[2 * w] * pb[j * l + 1];
                  s22 += pa[2 * w] * pb[j * l + 2];
                }
              pc[i * l] += s00;
              pc[i * l + 1] += s01;
              pc[i * l + 2] += s02;
              pc[(i + 1) * l] += s10;
              pc[(i + 1) * l + 1] += s11;
              pc[(i + 1) * l + 2] += s12;
              pc[(i + 2) * l] += s20;
              pc[(i + 2) * l + 1] += s21;
              pc[(i + 2) * l + 2] += s22;
            }
        }
      }
    else
      {
        for (m = 0, pc = c; m < l; m++, pc++)
          {
            for (i = 0, pb = b + m; i < h; i++)
              {
                double sum = 0;
                pa = a + i * w;
                for (j = 0; j < w; j++, pa++)
                  {
                    sum += pa[0] * pb[j * l];
                  }
                pc[i * l] += sum;
              }
          }
      }
}


int Jordan (double *a, double *b, double *c/*буферная строка*/, int n, int m, int p, int k, MPI_Comm G)
{
  int Nr = num_block_rows (n, m);
  int sender = find_sender (Nr, p);
  int RpP = rows_p_process (Nr, p, k);
  int l_h = n - (Nr - 1) * m;
  int up_bound_pivot = RpP; //верхняя граница для поиска главного элемента
  
  int a_height = (RpP - 1) * m;
  
  if (k == sender)
    a_height += l_h;
  else
    a_height += m;
  if (k == sender && m != l_h) //у процесса sender последняя строка урезанная
    {
      up_bound_pivot = RpP - 1;
    }
  
  double norm_a = norm (a, n, m, p, k);
  
  double *E = 0;
  double *F = 0;
  double *H = 0;
  
  double min_row;
  int non_sing;
  double min_norm;
  
  int loc_err = 0;
  int glob_err = 0;
  
  if (!(E = new double [m * m])
      || !(F = new double [m * m])
      || !(H = new double [m * m]))
    {
      loc_err = -3; //недостаточно памяти для хранения блока
    }
  MPI_Allreduce (&loc_err, &glob_err, 1, MPI_INT, MPI_MIN, G);
  
  if (glob_err < 0)
    {
      if (E) delete[] E;
      if (F) delete[] F;
      if (H) delete[] H;
      return glob_err;
    }
  //создание типа для передачи информации о главном элементе
  struct PivotMin piv = {0., 0, 0, 0};
  struct PivotMin piv_min = {0., 0, 0, 0};
  int t_c[4] = {1, 1, 1, 1};
  MPI_Aint t_d[4];
  MPI_Datatype t_t[4] = {MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT};
  MPI_Aint start_address, address;
  
  MPI_Address (&piv, &start_address);
  MPI_Address (&piv.min, &address);
  t_d[0] = address - start_address;
  MPI_Address (&piv.k, &address);
  t_d[1] = address - start_address;
  MPI_Address (&piv.min_row, &address);
  t_d[2] = address - start_address;
  MPI_Address (&piv.non_sing, &address);
  t_d[3] = address - start_address;
  
  MPI_Datatype MPI_PIVOT;
  
  MPI_Type_struct (4, t_c, t_d, t_t, &MPI_PIVOT);
  MPI_Type_commit (&MPI_PIVOT);
  
  MPI_Op MINPIV;
  MPI_Op_create (*pivot_func, 0, &MINPIV);
  
  for (int t = 0; t < Nr; t++) //проходим все столбцы
    {
      int start_row = t - k < 0 ? 0: ((t - k) % p? (t - k) / p + 1: (t - k) / p); //локальный номер ряда, начиная с которого ищем главный элемент
      int k_t = t % p;//номер процесса обрабатывающего эту строку
      int k_t_row = (t - k_t) / p;//номер этой строки в процессе k_t
      if (t == Nr - 1 && k == sender) //обрабатываем последний столбец и находимся в процессе sender
        up_bound_pivot = RpP; //повысили на +1 в случае урезанной последней строки
      
      min_norm = 0;
      non_sing = 0;
      min_row = start_row;
      
      //поиск главного элемента среди своих строк
      for (int i = start_row; i < up_bound_pivot; i++)
        {
          int pivot_h = 0;
          int pivot_w = 0;
          double bl_norm;
          
          get (a, E, i, t, a_height, n, m, pivot_h, pivot_w);
          int inv_res = inverse_block (E, F, norm_a, pivot_h);
          //F - хранит обратный к E
          
          if (!inv_res)
            {
              bl_norm = block_norm (F, pivot_h, pivot_w);
          
              if (!non_sing)
                {
                  min_row = i;
                  min_norm = bl_norm;
                }
              non_sing = 1;
          
              if (bl_norm < min_norm)
                {
                  min_norm = bl_norm;
                  min_row = i;
                }
            }
        }
          /*if (k == 0)
            printf ("process: %d, row: %d, block: %d, pivot_norm: %f\n", k, t, i, bl_norm);*/
      piv.min = min_norm;
      piv.k = k;
      piv.min_row = min_row;
      piv.non_sing = non_sing;
      
      MPI_Allreduce (&piv, &piv_min, 1, MPI_PIVOT, MINPIV, G);
      if (piv_min.non_sing == 0)//вырожденная матрица
        {
          delete []E;
          delete []H;
          delete []F;
          MPI_Type_free (&MPI_PIVOT);
          MPI_Op_free (&MINPIV);
          return -2; //сюда все процессы зашли одновременно 
        }
      /*if (k == 0)
        {
          printf ("process: %d, norm: %f, row: %d, non_sing: %d\n", piv_min.k, piv_min.min, piv_min.min_row, piv_min.non_sing);
        }*/
      int min_k = piv_min.k;
      int min_row = piv_min.min_row;
      int h_min_row = (min_k + p * min_row) == (Nr - 1)? l_h: m; //высота строки с главным элементом
      int w_min_row_a = n - t * m; //ширина строки с главным элементом
      double *pa, *pb, *pc;
      if (k == min_k)
      {
        gather_row (a, b, c, n, m, t, k_t_row, p, k);
      }
      MPI_Bcast (c, w_min_row_a * h_min_row + m * n, MPI_DOUBLE, min_k, G);
      //перестановка текщей строки и строки с главным элементом
       //после этого шага в матрице С уже стоит строка с главным элементом
      if (k_t == min_k)
        {
          if (k == min_k && min_row != k_t_row)
          {
            pa = a + t * m;
            for (int j = 0; j < w_min_row_a; j++, pc++, pa++)
              {
                for (int i = 0; i < h_min_row; i++)
                  pa[(min_row * m + i) * n] = pa [(k_t_row * m + i) * n]; //min_row встает на место t
              }
            pb = b;
            for (int j = 0; j < n; j++, pc++, pb++)
              {
                for (int i = 0; i < h_min_row; i++)
                  pb[(i + min_row * m) * n] = pb [(i + k_t_row * m) * n]; //min_row встает на место t
              }
            }
        }
      else
        {
          if (k == k_t)
            {
              MPI_Send (a + k_t_row * m * n, m * n, MPI_DOUBLE, min_k, 0, G);
              MPI_Send (b + k_t_row * m * n, m * n, MPI_DOUBLE, min_k, 0, G);
            }
          if (k == min_k)
            {
              MPI_Status status;
              MPI_Recv (a + min_row * m * n, m * n, MPI_DOUBLE, k_t, 0, G, &status);
              MPI_Recv (b + min_row * m * n, m * n, MPI_DOUBLE, k_t, 0, G, &status);
            }
        }
      int h_piv = 0, w_piv = 0; //высота и длина главного элемента
      get (c, E, 0, 0, h_min_row, w_min_row_a, m, h_piv, w_piv);
      inverse_block (E, H, norm_a, h_piv); //в H хранится обратный эелемент к главтому
      
      for (int j = 1; j < Nr - t; j++) //домножаем строку в с (часть А)
        {
          int bl_h = 0, bl_w = 0;
          get (c, F, 0, j, h_min_row, w_min_row_a, m, bl_h, bl_w);
          mult_block (H, F, E, h_piv, w_piv, bl_w); //в F хранится результат умножения
          set (E, c, 0, j, h_min_row, w_min_row_a, m);
          if (k == k_t) //в случае если строкв с главным элементом не совпадает с текущей строкой (это точно не случай, когда t = Nr - 1)
            {
              set (E, a, k_t_row, j + t, a_height, n, m);
            }
            
        }
      
      for (int j = 0; j < Nr; j++) //домножаем строку в с (часть B)
        {
          int bl_h = 0, bl_w = 0;
          get (c + h_min_row * w_min_row_a, F, 0, j, h_min_row, n, m, bl_h, bl_w);
          mult_block (H, F, E, h_piv, w_piv, bl_w); //в F хранится результат умножения
          set (E, c + h_min_row * w_min_row_a, 0, j, h_min_row, n, m);
          if (k == k_t) //в случае если строкв с главным элементом не совпадает с текущей строкой (это точно не случай, когда t = Nr - 1)
            {
              set (E, b, k_t_row, j, a_height, n, m);
            }
        }
        
        
        //в E хранится элемент A (i, t)
        //в H хранится элемень С(i)
        //на этом этапе остралось произвести вычетания из строк главной строки с главным элементом
        for (int i = 0; i < RpP; i++)
          {
            if (i == k_t_row && k == k_t)//строку t матрицы мы уже посчитали
              {
                continue;
              }
            int lead_bl_h = 0, lead_bl_w = 0; //размерность ведущего элемента строки i (lead_bl_h)
            get (a, E, i, t, a_height, n, m, lead_bl_h, lead_bl_w);
            //в матрице H хранится элемент A(i, j)
            //в матрице F хранится элемент A(t, j) == C(j)
            //домножкние i-ой строки матрицы А
            for (int j = t + 1; j < Nr; j++)
              {
                int bl_w_a, bl_h_a; //размерность элемента A(i, j)
                get (a, H, i, j, a_height, n, m, bl_h_a, bl_w_a);
                int bl_w_c = 0, bl_h_c = 0; //размерность элемента С(j) bl_w_a == bl_w_c
                get (c, F, 0, j - t, h_min_row, w_min_row_a, m, bl_h_c, bl_w_c);
                mult_substr_block (E, F, H, lead_bl_h, lead_bl_w, bl_w_c);
                set (H, a, i, j, a_height, n, m);
              }
            for (int j = 0; j < Nr; j++)
              {
                int bl_w_b, bl_h_b; //размерность элемента B(i, j)
                get (b, H, i, j, a_height, n, m, bl_h_b, bl_w_b);
                int bl_w_c = 0, bl_h_c = 0; //размерность элемента С(j) bl_w_b == bl_w_c
                get (c + h_min_row * w_min_row_a, F, 0, j, h_min_row, n, m, bl_h_c, bl_w_c);
                mult_substr_block (E, F, H, lead_bl_h, lead_bl_w, bl_w_c);
                set (H, b, i, j, a_height, n, m);
              }
          }
        
    }
  delete []E;
  delete []H;
  delete []F;
  MPI_Type_free (&MPI_PIVOT);
  MPI_Op_free (&MINPIV);
  return 0;
  
}

void minus_i (double *d, int n, int m, int p, int k)
{
  int Nr = num_block_rows (n, m);
  int sender = find_sender (Nr, p);
  int RpP = rows_p_process (Nr, p, k);
  int loc_i = 0;
  int loc_j = k * m;
  int up_bound = m;
  int l_h = k == sender? n - (Nr - 1) * m: m;
  
  for (int i = 0; i < RpP; i++, loc_i += m, loc_j += p * m)
    {
      if (i == RpP - 1)
        up_bound = l_h;
      for (int u = 0; u < up_bound; u++)
        d[(u + loc_i) * n + loc_j + u] -= 1;
    }
    
}
